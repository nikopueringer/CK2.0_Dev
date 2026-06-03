"""Frozen foundation-model wrappers for CorridorKey v3.

Produces per-token features at /64 of the input resolution:
  • C-RADIOv4-SO400M: input (H/4, W/4), patch 16 → /64 grid.
  • MoGe-ViT-B:     input (7H/32, 7W/32), patch 14 → /64 grid.

Outputs are normalized per-channel via precomputed mean/std (loaded via
`load_stats`); when stats aren't loaded, the normalization is pass-through
(mean=0, std=1).

Stats are computed offline by `scripts/calibrate_foundations.py` against
our training distribution and saved to a .pt file.
"""

import importlib.util
import os
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MoGe loader
# ---------------------------------------------------------------------------


def build_moge_vitb(device='cuda', checkpoint='Ruicheng/moge-2-vitb-normal'):
    from moge.model.v2 import MoGeModel
    m = MoGeModel.from_pretrained(checkpoint)
    m = m.to(device).eval()
    try:
        m.enable_pytorch_native_sdpa()
    except Exception:
        pass
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ---------------------------------------------------------------------------
# C-RADIO loader
# ---------------------------------------------------------------------------


def _load_local_cradio_so400m(repo_path: Path):
    """Load bundled C-RADIO remote-code files without HF dynamic-module cache.

    The exported runtime ships a full local HuggingFace snapshot. Loading the
    class directly avoids any dependency on the user's global
    ~/.cache/huggingface/modules state while still using the official C-RADIO
    implementation and safetensors weights from the snapshot.
    """
    hf_model_path = repo_path / "hf_model.py"
    if not hf_model_path.is_file():
        raise FileNotFoundError(f"C-RADIO snapshot is missing {hf_model_path}")

    package_name = "_corridorkey_cradio_snapshot"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__file__ = str(repo_path / "__init__.py")
        package.__path__ = [str(repo_path)]
        sys.modules[package_name] = package

    module_name = f"{package_name}.hf_model"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, hf_model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import bundled C-RADIO module at {hf_model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    config = module.RADIOConfig.from_pretrained(
        str(repo_path), local_files_only=True,
    )
    return module.RADIOModel.from_pretrained(
        str(repo_path), config=config, local_files_only=True,
    )


def build_cradio_so400m(repo: str = 'nvidia/C-RADIOv4-SO400M', device='cuda'):
    """Build frozen C-RADIOv4-SO400M.

    C-RADIO's internal input conditioner expects RGB in [0, 1] unless
    `make_preprocessor_external()` has been called. We keep the conditioner
    internal so callers can feed the same range used by the rest of this
    pipeline.
    """
    repo_path = Path(repo)
    if repo_path.is_dir():
        m = _load_local_cradio_so400m(repo_path)
    else:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(
            repo, trust_remote_code=True, local_files_only=True,
        )
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval().to(device)


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------


class FrozenFoundations(nn.Module):
    """Frozen C-RADIO + MoGe/RVM feature extractor producing /64 features.

    Stats normalization: outputs are normalized channel-wise with
        normalized = (x - mean) / std
    using precomputed per-channel buffers (`cradio_mean`, `cradio_std`,
    `moge_mean`, `moge_std`, `rvm_mean`, `rvm_std`). When stats file hasn't been loaded the
    buffers hold (0, 1) so normalization is the identity. Stats are
    captured by `scripts/calibrate_foundations.py` over a combined sample
    of training distribution + ImageNet.
    """

    def __init__(self, device='cuda',
                 use_moge=True,
                 use_cradio: bool = False,
                 cradio_repo: str = 'nvidia/C-RADIOv4-SO400M',
                 moge_checkpoint: str = 'Ruicheng/moge-2-vitb-normal',
                 rvm_variant: str = 'small',
                 rvm_ckpt: str = None,
                 stats_path: str = None):
        super().__init__()
        self.use_moge = use_moge
        self.use_cradio = use_cradio
        self.cradio_repo = cradio_repo if use_cradio else None
        # RVM is implicitly enabled by passing --rvm_ckpt.
        self.use_rvm = rvm_ckpt is not None and os.path.isfile(str(rvm_ckpt))
        self.rvm_variant = rvm_variant if self.use_rvm else None

        self.moge_checkpoint = str(moge_checkpoint) if use_moge else None
        self.moge = (
            build_moge_vitb(device=device, checkpoint=self.moge_checkpoint)
            if use_moge else None
        )
        self.cradio = build_cradio_so400m(
            repo=cradio_repo, device=device,
        ) if use_cradio else None

        # RVM (Recurrent Video MAE from google-deepmind/representations4d).
        # Stateful across frames; input in [0, 1] (no ImageNet norm).
        if self.use_rvm:
            from .rvm import RVMVideoSiamMAE, load_rvm_npz
            self.rvm = RVMVideoSiamMAE(variant=rvm_variant).to(device)
            load_rvm_npz(self.rvm, rvm_ckpt, strict_load=True, verbose=True)
            for p in self.rvm.parameters():
                p.requires_grad_(False)
            self.rvm.eval()
        else:
            self.rvm = None

        # Per-channel output stats (buffers for state-dict persistence).
        # Shapes: (C,). Default 0/1 → pass-through.
        moge_c = 768 if use_moge else 0
        cradio_c = self._infer_cradio_dim() if use_cradio else 0
        rvm_c = self.rvm.embed_dim if (self.use_rvm and self.rvm is not None) else 0
        self.register_buffer('moge_mean', torch.zeros(max(moge_c, 1)))
        self.register_buffer('moge_std',  torch.ones(max(moge_c, 1)))
        self.register_buffer('rvm_mean',  torch.zeros(max(rvm_c, 1)))
        self.register_buffer('rvm_std',   torch.ones(max(rvm_c, 1)))
        if use_cradio:
            self.register_buffer('cradio_mean', torch.zeros(cradio_c))
            self.register_buffer('cradio_std',  torch.ones(cradio_c))

        if stats_path is not None and os.path.isfile(stats_path):
            self.load_stats(stats_path)

        # Make sure the wrapper's own buffers land on the right device.
        self.to(device)

    def train(self, mode: bool = True):
        """Keep frozen feature extractors in eval mode at all times."""
        super().train(mode)   # sets self.training = mode
        if self.moge is not None:
            self.moge.eval()
            for p in self.moge.parameters():
                p.requires_grad_(False)
        if self.rvm is not None:
            self.rvm.eval()
            for p in self.rvm.parameters():
                p.requires_grad_(False)
        if self.cradio is not None:
            self.cradio.eval()
            for p in self.cradio.parameters():
                p.requires_grad_(False)
        return self

    def load_stats(self, path: str):
        """Load per-channel mean/std from a .pt file produced by
        `scripts/calibrate_foundations.py`. File format:
            {'cradio_mean': (C_c,), 'cradio_std': (C_c,),
             'moge_mean': (C_m,), 'moge_std': (C_m,),
             'meta': {...}}  (everything else is metadata for logging)
        Silently skips any key we don't have a buffer for."""
        stats = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('moge_mean', 'moge_std', 'rvm_mean', 'rvm_std',
                    'cradio_mean', 'cradio_std'):
            if key in stats and hasattr(self, key):
                buf = getattr(self, key)
                val = stats[key].to(buf.device, buf.dtype)
                if val.shape != buf.shape:
                    print(f"[FrozenFoundations] {key}: stats shape "
                          f"{tuple(val.shape)} != buffer shape {tuple(buf.shape)} — skipping")
                    continue
                buf.copy_(val)
        print(f"[FrozenFoundations] loaded stats from {path}")

    @torch.no_grad()
    def forward(self, rgb_moge=None, rgb_cradio=None, rgb_rvm=None,
                rvm_state=None):
        """C-RADIO and MoGe consume (B·T, 3, ...) 4D tensors (per-frame IID).
        RVM consumes (B, T, 3, H_rvm, W_rvm) 5D because it is recurrent across T.

        Args:
            rgb_cradio: (B·T, 3, H/4, W/4) — if C-RADIO enabled.
            rgb_moge: (B·T, 3, 7H/32, 7W/32) — if MoGe enabled.
            rgb_rvm:  (B, T, 3, H_rvm, W_rvm) — if RVM enabled. Typically
                      the same H/4 input used by C-RADIO.
            rvm_state: optional (B, 1+N_rvm, D_rvm) carry state for RVM.
                      At training time, pass None (fresh per batch).

        Returns:
            Dict with channel-normalized features:
              'moge':      (B·T, C_m, H/64, W/64)
              'cradio':    (B·T, C_c, H/64, W/64)
              'rvm':       (B·T, C_r, h_rvm, w_rvm)  (grid at RVM input / 16)
              'rvm_state': (B, 1+N_rvm, D_rvm)  final state, for streaming
                           inference across windows (ignored in training).
        """
        out = {}

        if self.cradio is not None:
            assert rgb_cradio is not None, "C-RADIO enabled but rgb_cradio not provided"
            try:
                cradio_out = self.cradio(rgb_cradio, feature_fmt='NCHW')
            except TypeError:
                cradio_out = self.cradio(rgb_cradio)
            cradio_feat = cradio_out.features
            if cradio_feat.dim() == 3:
                Bc, n_tok, Cc = cradio_feat.shape
                patch = int(getattr(self.cradio, 'patch_size', 16))
                hc, wc = rgb_cradio.shape[-2] // patch, rgb_cradio.shape[-1] // patch
                if n_tok != hc * wc:
                    raise RuntimeError(
                        f"C-RADIO token count {n_tok} != expected grid {hc}x{wc}")
                cradio_feat = cradio_feat.transpose(1, 2).reshape(Bc, Cc, hc, wc)
            cradio_feat = cradio_feat.float()
            mean = self.cradio_mean.to(
                cradio_feat.device, cradio_feat.dtype,
            ).view(1, -1, 1, 1)
            std = self.cradio_std.to(
                cradio_feat.device, cradio_feat.dtype,
            ).view(1, -1, 1, 1)
            cradio_feat = (cradio_feat - mean) / std.clamp(min=1e-6)
            out['cradio'] = cradio_feat

        if self.moge is not None:
            assert rgb_moge is not None, "MoGe enabled but rgb_moge not provided"
            Bm, _, Hm, Wm = rgb_moge.shape
            base_h, base_w = Hm // 14, Wm // 14
            moge_feat, _ = self.moge.encoder(
                rgb_moge, base_h, base_w, return_class_token=True,
            )
            moge_feat = moge_feat.float()
            mean = self.moge_mean.to(moge_feat.device, moge_feat.dtype).view(1, -1, 1, 1)
            std  = self.moge_std.to(moge_feat.device, moge_feat.dtype).view(1, -1, 1, 1)
            moge_feat = (moge_feat - mean) / std.clamp(min=1e-6)
            out['moge'] = moge_feat

        # RVM is allowed to be skipped when rgb_rvm is None even if the
        # model is enabled — useful when the caller pre-computes RVM
        # features externally (e.g. inference's full-clip stateful pass)
        # and supplies them to stage1 directly. In that case foundations
        # shouldn't also run RVM inline.
        if self.rvm is not None and rgb_rvm is not None:
            # rgb_rvm: (B, T, 3, H, W). No mean/std norm — RVM wants [0, 1].
            rvm_feat, new_state = self.rvm(rgb_rvm, state=rvm_state)   # (B, T, D, h, w)
            B_r, T_r, D_r, h_r, w_r = rvm_feat.shape
            rvm_feat = rvm_feat.reshape(B_r * T_r, D_r, h_r, w_r).float()
            mean = self.rvm_mean.to(rvm_feat.device, rvm_feat.dtype).view(1, -1, 1, 1)
            std  = self.rvm_std.to(rvm_feat.device, rvm_feat.dtype).view(1, -1, 1, 1)
            rvm_feat = (rvm_feat - mean) / std.clamp(min=1e-6)
            out['rvm'] = rvm_feat
            out['rvm_state'] = new_state

        return out

    def _infer_cradio_dim(self):
        if self.cradio is None:
            return 0
        for obj in (self.cradio, getattr(self.cradio, 'radio_model', None),
                    getattr(self.cradio, 'model', None)):
            dim = getattr(obj, 'embed_dim', None) if obj is not None else None
            if dim is not None:
                return int(dim)
        raise AttributeError("Unable to infer C-RADIO embed_dim")

    @property
    def moge_dim(self):
        return 768 if self.use_moge else 0

    @property
    def cradio_dim(self):
        if not self.use_cradio:
            return 0
        return self._infer_cradio_dim()

    @property
    def rvm_dim(self):
        if not self.use_rvm or self.rvm is None:
            return 0
        return self.rvm.embed_dim

    @property
    def vfm_dim(self):
        """Concatenated foundation channel count."""
        return self.cradio_dim + self.moge_dim + self.rvm_dim
