from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import color_utils as cu
from .model import CorridorKeyV3
from .video_io import (
    ThreadedVideoWriter,
    WriterConfig,
    apply_linear_adjust,
    frame_paths,
    has_linear_adjust,
    iter_frame_dir_chunks,
    iter_video_frame_chunks,
    load_initial_hint_frame,
    make_checker,
    make_hint_reader,
    none_path,
    probe_frame_dir_meta,
    probe_video_meta,
    raw_clip_name,
    safe_remove,
    taper_weights,
    window_starts,
    SequentialFrameDirWindowReader,
    SequentialVideoWindowReader,
)


@dataclass
class RuntimePaths:
    package_root: Path
    weights: Path
    foundation_stats: Path
    rvm_ckpt: Path
    moge_checkpoint: Path
    cradio_dir: Path

    @staticmethod
    def _resolve_user_or_package_path(root: Path, value, default: str) -> Path:
        if value is None:
            return (root / default).resolve()
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return cwd_path
        return (root / path).resolve()

    @classmethod
    def from_package(cls, package_root: str | os.PathLike,
                     weights: str | os.PathLike | None = None):
        root = Path(package_root).resolve()
        return cls(
            package_root=root,
            weights=cls._resolve_user_or_package_path(
                root, weights, "weights/corridorkey_v2.pt"),
            foundation_stats=(root / "weights/foundation_stats.pt").resolve(),
            rvm_ckpt=(root / "vfms/rvm/pretrain_rvm_small16_256_204031069.npz").resolve(),
            moge_checkpoint=(root / "vfms/moge/model.pt").resolve(),
            cradio_dir=(root / "vfms/cradio/C-RADIOv4-SO400M").resolve(),
        )

    def validate(self) -> None:
        missing = [
            p for p in (
                self.weights, self.foundation_stats, self.rvm_ckpt,
                self.moge_checkpoint, self.cradio_dir,
            )
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "missing runtime asset(s): " + ", ".join(str(p) for p in missing))


@dataclass
class InferenceConfig:
    hann_chunk: int = 80
    hann_stride: int = 40
    carry_hint: bool = True
    hint_quality: float = 0.95
    carry_hint_quality: float = 0.95
    shape_quantum: int = 128
    model_scale: float = 1.0
    num_frames: int = -1
    start_mode: str = "begin"
    seed: int = 0
    low_vram: bool = False
    low_vram_foundation_chunk: int = 4
    low_vram_mlp_chunk_tokens: int = 65536
    low_vram_enc_frame_chunk: int = 1
    low_vram_swin_window_batch: int = 16
    low_vram_full_attn_query_chunk_tokens: int = 2048
    stage2_batch: int = 4
    outputs: set[str] = field(default_factory=lambda: {"alpha", "fg"})
    temp_dir: Path = Path("/tmp/corridorkey_v2")
    linear_brightness: float = 1.0
    linear_contrast: float = 1.0
    linear_contrast_pivot: float = 0.18
    frame_dir_fps: float = 24.0
    frame_dir_linear: bool = False
    cutout_linear: bool = False
    despill_strength: float = 0.5


def prepare_runtime_environment(paths: RuntimePaths) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    moge_root = paths.package_root / "third_party/MoGe"
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))


def _resolve_package_path(paths: RuntimePaths, value: str | None) -> str | None:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = paths.package_root / p
    return str(p.resolve())


def load_model(paths: RuntimePaths, device: str = "cuda") -> CorridorKeyV3:
    paths.validate()
    prepare_runtime_environment(paths)
    ckpt = torch.load(paths.weights, map_location="cpu", weights_only=False, mmap=True)
    if ckpt.get("format") != "corridorkey_v2_runtime_checkpoint":
        raise RuntimeError(f"{paths.weights} is not a CorridorKey v2 runtime checkpoint")
    cfg = dict(ckpt["model_config"])
    cfg["cradio_repo"] = str(paths.cradio_dir)
    cfg["moge_checkpoint"] = str(paths.moge_checkpoint)
    cfg["rvm_ckpt"] = str(paths.rvm_ckpt)
    cfg["foundation_stats_path"] = str(paths.foundation_stats)
    cfg["gradient_checkpoint"] = False
    cfg["device"] = device
    cfg["stage1_recurrent_64_sites"] = ()

    print(f"Loading CorridorKey v2 runtime weights: {paths.weights}")
    model = CorridorKeyV3(**cfg)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    train_missing = [k for k in missing if not k.startswith("foundations.")]
    train_unexpected = [k for k in unexpected if not k.startswith("foundations.")]
    if train_missing or train_unexpected:
        raise RuntimeError(
            "runtime checkpoint did not match model: "
            f"missing={train_missing[:8]} unexpected={train_unexpected[:8]}")
    print(f"  loaded weights; omitted frozen foundation tensors={len(missing)}")
    model.eval()
    del ckpt
    return model


def configure_low_vram_inference(model, cfg: InferenceConfig):
    model.stage1.low_vram_enc_frame_chunk = max(0, int(cfg.low_vram_enc_frame_chunk))
    for module in model.modules():
        name = module.__class__.__name__
        if name in ("SwiGLU", "FusionLinSwiglu"):
            module.inference_chunk_tokens = max(0, int(cfg.low_vram_mlp_chunk_tokens))
        if name == "SwinAttention3D":
            module.inference_window_batch = max(0, int(cfg.low_vram_swin_window_batch))
        if name == "CSwinAttention3D":
            module.inference_strip_batch = max(0, int(cfg.low_vram_swin_window_batch))
        if name in ("FullSpatialAttention2D", "FullAttention3D"):
            module.inference_query_chunk_tokens = max(
                0, int(cfg.low_vram_full_attn_query_chunk_tokens))


def _build_foundation_inputs(rgb_bt: torch.Tensor):
    B, T, _, H, W = rgb_bt.shape
    flat = rgb_bt.reshape(B * T, 3, H, W)
    rgb_radio = F.interpolate(flat, size=(H // 4, W // 4), mode="area")
    H_moge, W_moge = (H * 7) // 32, (W * 7) // 32
    rgb_moge = F.interpolate(flat, size=(H_moge, W_moge), mode="area")
    return (
        rgb_radio.reshape(B, T, 3, H // 4, W // 4),
        rgb_moge.reshape(B, T, 3, H_moge, W_moge),
    )


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


@torch.no_grad()
def precompute_rvm_to_memmap(model, source, meta, rvm_path: Path,
                             *, chunk: int, reader_kind: str):
    assert model.foundations.rvm is not None
    device = next(model.foundations.rvm.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    H_e, W_e = H // 4, W // 4
    D = model.foundations.rvm.embed_dim
    h_e, w_e = H_e // 16, W_e // 16
    print(f"    RVM continuous prepass: T={T_all}, chunk={chunk}, "
          f"input={W_e}x{H_e}, grid={w_e}x{h_e}")
    out = np.memmap(rvm_path, dtype=np.float16, mode="w+",
                    shape=(T_all, D, h_e, w_e))
    mean = model.foundations.rvm_mean.view(1, 1, -1, 1, 1).float().cpu()
    std = model.foundations.rvm_std.view(1, 1, -1, 1, 1).float().cpu().clamp(min=1e-6)
    if reader_kind == "video":
        chunks = iter_video_frame_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    else:
        chunks = iter_frame_dir_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    state = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            Tc = display.shape[0]
            rgb = display.unsqueeze(0).to(device, non_blocking=True).float()
            rgb_rvm = F.interpolate(
                rgb.reshape(Tc, 3, H, W), size=(H_e, W_e),
                mode="bilinear", align_corners=False,
            ).reshape(1, Tc, 3, H_e, W_e).clamp(0.0, 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats, state = model.foundations.rvm(rgb_rvm, state=state)
            feats_cpu = (feats.float().cpu() - mean) / std
            out[t0:t0 + Tc] = feats_cpu[0].numpy().astype(np.float16)
            del rgb, rgb_rvm, feats, feats_cpu, display
    out.flush()
    del out
    torch.cuda.empty_cache()


@torch.no_grad()
def precompute_foundations_to_memmap(model, source, meta, prefix: Path,
                                     *, chunk: int, reader_kind: str):
    device = next(model.foundations.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    h64, w64 = H // 64, W // 64
    C_cradio = int(getattr(model.stage1, "cradio_dim", 0) or 0)
    C_moge = int(getattr(model.stage1, "moge_dim", 0) or 0)
    if C_moge <= 0:
        raise RuntimeError("MoGe is required for this model")
    paths = {
        "moge": str(prefix) + "_moge.f32",
        "moge_shape": (T_all, C_moge, h64, w64),
    }
    moge_out = np.memmap(paths["moge"], dtype=np.float32, mode="w+",
                         shape=paths["moge_shape"])
    cradio_out = None
    if C_cradio > 0:
        paths["cradio"] = str(prefix) + "_cradio.f32"
        paths["cradio_shape"] = (T_all, C_cradio, h64, w64)
        cradio_out = np.memmap(paths["cradio"], dtype=np.float32, mode="w+",
                               shape=paths["cradio_shape"])
    print(f"    frozen VFM prepass: T={T_all}, grid={w64}x{h64}, chunk={chunk}")
    if reader_kind == "video":
        chunks = iter_video_frame_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    else:
        chunks = iter_frame_dir_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            Tc = display.shape[0]
            rgb = display.unsqueeze(0).to(device, non_blocking=True)
            rgb_radio, rgb_moge = _build_foundation_inputs(rgb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ff = model.foundations(
                    rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                    rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                )
            moge_out[t0:t0 + Tc] = ff["moge"].float().cpu().numpy()
            if cradio_out is not None:
                cradio_out[t0:t0 + Tc] = ff["cradio"].float().cpu().numpy()
            del rgb, rgb_radio, rgb_moge, ff, display
    moge_out.flush()
    del moge_out
    if cradio_out is not None:
        cradio_out.flush()
        del cradio_out
    torch.cuda.empty_cache()
    return paths


def _phase1_setup(model):
    model.foundations.cuda()
    model.stage1.cuda()
    model.stage2.cpu()
    torch.cuda.empty_cache()


def _phase2_setup(model):
    model.foundations.cpu()
    model.stage1.cpu()
    torch.cuda.empty_cache()
    model.stage2.cuda()
    torch.cuda.synchronize()


@torch.no_grad()
def hann_stage1_stream(model, source, meta, *, reader_cls, reader_kind: str,
                       chunk: int, stride: int, baton_path: Path,
                       initial_hint, initial_hint_quality: float,
                       full_hint_video_path: str | None,
                       full_hint_quality: float,
                       rvm_path: Path | None,
                       foundation_paths,
                       carry_hint_from_aux: bool,
                       carry_hint_quality: float,
                       on_baton_frames):
    device = next(model.stage1.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    d_16 = model.stage1.d_16
    h16, w16 = H // 16, W // 16
    starts = window_starts(T_all, chunk, stride)
    overlap = max(0, chunk - stride)
    baton_fp16 = None
    if on_baton_frames is None:
        baton_fp16 = np.memmap(baton_path, dtype=np.float16, mode="w+",
                               shape=(T_all, d_16, h16, w16))
    live_acc = {}
    live_w = {}
    print(f"    Hann stage1: chunk={chunk}, stride={stride}, "
          f"windows={len(starts)}, baton_grid={w16}x{h16}")

    rvm_map = None
    if rvm_path is not None:
        D = model.foundations.rvm.embed_dim
        rvm_map = np.memmap(rvm_path, dtype=np.float16, mode="r",
                            shape=(T_all, D, H // 64, W // 64))
    cradio_map = moge_map = None
    if foundation_paths is not None:
        moge_map = np.memmap(foundation_paths["moge"], dtype=np.float32,
                             mode="r", shape=tuple(foundation_paths["moge_shape"]))
        if model.stage1.cradio_dim > 0:
            cradio_map = np.memmap(foundation_paths["cradio"], dtype=np.float32,
                                   mode="r", shape=tuple(foundation_paths["cradio_shape"]))

    reader = reader_cls(str(source), meta)
    hint_reader = make_hint_reader(full_hint_video_path, meta)
    carry_cache = {}
    stage2_frame_cache = {}

    def flush_before(t_limit: int):
        ready = []
        for k in sorted(k for k in live_acc if k < t_limit):
            frame = (live_acc.pop(k) / max(live_w.pop(k), 1e-6)).astype(np.float16)
            if on_baton_frames is None:
                baton_fp16[k] = frame
            else:
                model_frame = stage2_frame_cache.pop(k)
                ready.append((k, frame, model_frame))
        if ready:
            on_baton_frames(ready)

    try:
        for i, t0 in enumerate(starts):
            t1 = min(t0 + chunk, T_all)
            Tc = t1 - t0
            model_chunk, display_chunk, review_chunk = reader.window(t0, t1)
            hint_cpu = torch.zeros(1, Tc, 1, H, W, dtype=torch.bfloat16)
            hint_quality_cpu = torch.zeros(1, Tc, dtype=torch.bfloat16)
            hint_present = torch.zeros(1, Tc, dtype=torch.bool)
            if hint_reader is not None:
                hint_cpu[0] = hint_reader.window(t0, t1)
                hint_quality_cpu.fill_(float(full_hint_quality))
                hint_present.fill_(True)
            for j, t_abs in enumerate(range(t0, t1)):
                if t_abs in carry_cache:
                    hint_cpu[0, j] = carry_cache[t_abs]
                    hint_quality_cpu[0, j] = float(carry_hint_quality)
                    hint_present[0, j] = True
            if initial_hint is not None and t0 == 0:
                hint_cpu[0, 0] = initial_hint.to(dtype=hint_cpu.dtype)
                hint_quality_cpu[0, 0] = float(initial_hint_quality)
                hint_present[0, 0] = True
            if not bool(hint_present.any().item()):
                hint_quality_cpu.fill_(-1.0)

            if on_baton_frames is not None:
                for j, t_abs in enumerate(range(t0, t1)):
                    stage2_frame_cache[t_abs] = model_chunk[j]

            rgb_c = model_chunk.unsqueeze(0).to(device, non_blocking=True)
            hint_c = hint_cpu.to(device, non_blocking=True)
            hint_quality_c = hint_quality_cpu.to(device, non_blocking=True)
            ff = rgb_radio = rgb_moge = None
            if foundation_paths is None:
                rgb_found = display_chunk.unsqueeze(0).to(device, non_blocking=True)
                rgb_radio, rgb_moge = _build_foundation_inputs(rgb_found)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ff = model.foundations(
                        rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                        rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                    )
                cradio_feat = ff["cradio"].reshape(1, Tc, -1, H // 64, W // 64)
                moge_feat = ff["moge"].reshape(1, Tc, -1, H // 64, W // 64)
            else:
                moge_feat = torch.from_numpy(np.asarray(moge_map[t0:t1]).copy())
                moge_feat = moge_feat.unsqueeze(0).to(device, non_blocking=True).float()
                cradio_feat = torch.from_numpy(np.asarray(cradio_map[t0:t1]).copy())
                cradio_feat = cradio_feat.unsqueeze(0).to(device, non_blocking=True).float()
            rvm_feat = None
            if rvm_map is not None:
                rvm_feat = torch.from_numpy(np.asarray(rvm_map[t0:t1]).copy())
                rvm_feat = rvm_feat.unsqueeze(0).to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s1 = model.stage1(
                    rgb_c, hint_c, moge_feat, cradio_feat=cradio_feat,
                    rvm_feat=rvm_feat, t_stride=1.0,
                    hint_quality=hint_quality_c,
                )
            baton_gpu = s1["baton"][0].float()
            if carry_hint_from_aux:
                aux_alpha = s1["aux"][0, :, 0:1].float()
                aux_full = F.interpolate(
                    aux_alpha, size=(H, W), mode="bilinear",
                    align_corners=False).clamp(0.0, 1.0)
                aux_cpu = aux_full.to("cpu", dtype=torch.bfloat16)
                for j, t_abs in enumerate(range(t0, t1)):
                    carry_cache[t_abs] = aux_cpu[j].clone()
                next_start = starts[i + 1] if i + 1 < len(starts) else T_all
                for old_t in [k for k in carry_cache if k < next_start]:
                    del carry_cache[old_t]
                del aux_alpha, aux_full, aux_cpu
            w = taper_weights(Tc, overlap, is_first=(i == 0),
                              is_last=(i == len(starts) - 1),
                              device="cpu", dtype=torch.float32)
            weighted = (baton_gpu * w.to(device).view(Tc, 1, 1, 1)).to("cpu").numpy()
            w_np = w.numpy()
            for j, t_abs in enumerate(range(t0, t1)):
                if t_abs not in live_acc:
                    live_acc[t_abs] = weighted[j].copy()
                    live_w[t_abs] = float(w_np[j])
                else:
                    live_acc[t_abs] += weighted[j]
                    live_w[t_abs] += float(w_np[j])
            next_start = starts[i + 1] if i + 1 < len(starts) else T_all
            del s1, baton_gpu, weighted, ff, cradio_feat, moge_feat, rvm_feat
            del rgb_c, hint_c, hint_quality_c, hint_cpu, hint_quality_cpu
            del model_chunk, display_chunk, review_chunk
            flush_before(next_start)
            print(f"      window {i + 1}/{len(starts)} [{t0}:{t1}]", flush=True)
        torch.cuda.synchronize()
        flush_before(T_all + 1)
    finally:
        reader.close()
        if hint_reader is not None:
            hint_reader.close()

    if baton_fp16 is not None:
        baton_fp16.flush()
    del baton_fp16, live_acc, live_w, rvm_map, cradio_map, moge_map
    torch.cuda.empty_cache()


def _crop_native_torch(x: torch.Tensor, meta: dict) -> torch.Tensor:
    h, w = meta["model_native_hw"]
    pad_l, _, pad_t, _ = meta["pad"]
    return x[..., pad_t:pad_t + h, pad_l:pad_l + w]


def _resize_output_torch(x: torch.Tensor, meta: dict) -> torch.Tensor:
    if tuple(meta["model_native_hw"]) == tuple(meta["output_hw"]):
        return x
    raise RuntimeError(
        "optimized runtime render path requires native model_scale=1.0; "
        "non-native scaling would need explicit Lanczos output rendering")


def _linear_adjust_inverse_torch(linear: torch.Tensor, meta: dict) -> torch.Tensor:
    gain = float(meta.get("linear_brightness", 1.0))
    contrast = float(meta.get("linear_contrast", 1.0))
    if abs(gain - 1.0) <= 1e-6 and abs(contrast - 1.0) <= 1e-6:
        return linear
    pivot = float(meta.get("linear_contrast_pivot", 0.18))
    return (linear / gain - pivot) / contrast + pivot


def _despill_average_srgb_torch(fg: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    if strength <= 0.0:
        return fg
    r = fg[:, 0:1]
    g = fg[:, 1:2]
    b = fg[:, 2:3]
    spill = (g - (r + b) * 0.5).clamp(min=0.0)
    full = torch.cat([r + spill * 0.5, g - spill, b + spill * 0.5], dim=1)
    if strength >= 1.0:
        return full
    return fg * (1.0 - strength) + full * strength


def _srgb_to_bgr_uint16_cpu(x: torch.Tensor) -> np.ndarray:
    x = x.clamp(0.0, 1.0)
    x = x[:, [2, 1, 0]].permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return (x * 65535.0).astype(np.uint16)


def _srgb_to_bgra_uint16_cpu(rgb: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    rgb = rgb.clamp(0.0, 1.0)
    alpha = alpha.clamp(0.0, 1.0)
    bgra = torch.cat([rgb[:, [2, 1, 0]], alpha], dim=1)
    bgra = bgra.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return (bgra * 65535.0).astype(np.uint16)


def _render_predictions_torch(meta: dict, outputs: set[str],
                              checker: torch.Tensor,
                              logits: torch.Tensor) -> dict[str, np.ndarray]:
    """Render a stage-2 batch on GPU, returning uint16 BGR/BGRA NHWC arrays.

    This removes the previous per-frame numpy HDR decode, sRGB encode,
    composition, channel shuffle, and uint8 conversion hot path.
    """
    alpha_model = logits[:, 0:1].float()
    pa_model = _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0)
    despill_strength = float(meta.get("despill_strength", 0.5))

    pa = _resize_output_torch(pa_model, meta).clamp(0.0, 1.0)
    rendered: dict[str, np.ndarray] = {}
    if "alpha" in outputs:
        rendered["alpha"] = _srgb_to_bgr_uint16_cpu(pa.expand(-1, 3, -1, -1))
    if not (outputs & {"fg", "cutout", "checker"}):
        return rendered

    fg_model = logits[:, 1:4].float()
    pf_model = _crop_native_torch(fg_model, meta)
    pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
    pf_linear_pos = pf_linear.clamp(min=0.0)
    pf = _resize_output_torch(cu.linear_to_srgb(pf_linear_pos), meta).clamp(0.0, 1.0)
    if "fg" in outputs:
        rendered["fg"] = _srgb_to_bgr_uint16_cpu(pf)
    if "cutout" in outputs:
        pf_linear_despilled = _despill_average_srgb_torch(pf_linear_pos, strength=despill_strength)
        if meta.get("cutout_linear", False):
            pf_linear_despilled_resized = _resize_output_torch(pf_linear_despilled, meta).clamp(0.0, 1.0)
            rendered["cutout"] = _srgb_to_bgra_uint16_cpu(pf_linear_despilled_resized, pa)
        else:
            pf_despill = _despill_average_srgb_torch(pf, strength=despill_strength)
            rendered["cutout"] = _srgb_to_bgra_uint16_cpu(pf_despill, pa)
    if "checker" in outputs:
        checker_b = checker.to(device=pf.device, dtype=pf.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(checker_b)
        alpha = pa
        pf_despill = _despill_average_srgb_torch(pf, strength=despill_strength)
        comp_despill_lin = (
            cu.srgb_to_linear(pf_despill) * alpha + bg_lin * (1.0 - alpha))
        comp_despill = cu.linear_to_srgb(
            comp_despill_lin.clamp(min=0.0)).clamp(0.0, 1.0)
        rendered["checker"] = _srgb_to_bgr_uint16_cpu(comp_despill)
    return rendered


def run_inference(model, input_path: str, output_dir: str, cfg: InferenceConfig,
                  writer_cfg: WriterConfig, *, initial_hint_path: str | None,
                  hint_video_path: str | None):
    input_path = str(input_path)
    output_dir = str(output_dir)
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    is_dir = os.path.isdir(input_path)
    reader_kind = "frame_dir" if is_dir else "video"
    reader_cls = SequentialFrameDirWindowReader if is_dir else SequentialVideoWindowReader
    if is_dir:
        meta = probe_frame_dir_meta(
            input_path, num_frames=cfg.num_frames,
            shape_quantum=cfg.shape_quantum, fps=cfg.frame_dir_fps,
            exr_linear=cfg.frame_dir_linear, model_scale=cfg.model_scale)
    else:
        meta = probe_video_meta(
            input_path, num_frames=cfg.num_frames,
            shape_quantum=cfg.shape_quantum, native_resolution=True,
            start_mode=cfg.start_mode, seed=cfg.seed,
            model_scale=cfg.model_scale,
            linear_brightness=cfg.linear_brightness,
            linear_contrast=cfg.linear_contrast,
            linear_contrast_pivot=cfg.linear_contrast_pivot)
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    H_out, W_out = meta["output_hw"]
    meta["cutout_linear"] = cfg.cutout_linear
    meta["despill_strength"] = cfg.despill_strength
    fps = float(meta["native_fps"])
    clip_name = raw_clip_name(input_path)
    checker = make_checker(H_out, W_out)
    checker_t = torch.from_numpy(checker)
    writers = {}
    suffix = {
        "alpha": "alpha",
        "fg": "fg",
        "checker": f"checker_comp_despill{int(cfg.despill_strength * 100):03d}",
        "cutout": "cutout",
    }
    try:
        for name in sorted(cfg.outputs):
            if name not in suffix:
                raise ValueError(f"unsupported output choice: {name}")
            ext = ".mov" if name == "cutout" else ".mp4"
            channels = 4 if name == "cutout" else 3
            color_trc = "linear" if (name == "cutout" and cfg.cutout_linear) else "iec61966-2-1"
            writers[name] = ThreadedVideoWriter(
                os.path.join(output_dir, f"{clip_name}_{suffix[name]}{ext}"),
                fps, (W_out, H_out), writer_cfg, channels=channels, color_trc=color_trc)

        initial_hint = (
            load_initial_hint_frame(initial_hint_path, meta)
            if none_path(initial_hint_path) is not None else None
        )
        _phase1_setup(model)
        if cfg.low_vram:
            configure_low_vram_inference(model, cfg)
        rvm_path = cfg.temp_dir / f"{clip_name}_rvm_feat.f16"
        baton_path = cfg.temp_dir / f"{clip_name}_baton.f16"
        foundation_prefix = cfg.temp_dir / f"{clip_name}_foundation"
        for p in (
            rvm_path, baton_path, Path(str(foundation_prefix) + "_cradio.f32"),
            Path(str(foundation_prefix) + "_moge.f32"),
        ):
            safe_remove(p)
        if model.foundations.rvm is not None:
            precompute_rvm_to_memmap(
                model, input_path, meta, rvm_path,
                chunk=max(1, min(int(cfg.hann_chunk), 8)),
                reader_kind=reader_kind)
            rvm_feat_path = rvm_path
        else:
            rvm_feat_path = None
        foundation_paths = None
        if cfg.low_vram:
            foundation_paths = precompute_foundations_to_memmap(
                model, input_path, meta, foundation_prefix,
                chunk=max(1, int(cfg.low_vram_foundation_chunk)),
                reader_kind=reader_kind)
            model.foundations.cpu()
            model.stage2.cpu()
            model.stage1.cuda()
            torch.cuda.empty_cache()
        else:
            model.stage2.cuda()
            model.stage2.eval()
            torch.cuda.empty_cache()

        stage2_batch = max(1, int(cfg.stage2_batch))

        def write_prediction_batches(items):
            device = next(model.stage2.parameters()).device
            for offset in range(0, len(items), stage2_batch):
                sub = items[offset:offset + stage2_batch]
                rgb_t = torch.stack([item[2] for item in sub], dim=0)
                rgb_t = rgb_t.unsqueeze(0).to(device, non_blocking=True)
                baton_np = np.stack([item[1] for item in sub], axis=0)
                baton_t = torch.from_numpy(baton_np).unsqueeze(0)
                baton_t = baton_t.to(device, non_blocking=True).float()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.stage2(rgb_t, baton_t)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                rendered = _render_predictions_torch(
                    meta, cfg.outputs, checker_t, logits[0])
                for name, frames in rendered.items():
                    writers[name].write_many(frames)
                del logits, baton_t, rgb_t, baton_np, rendered

        hann_stage1_stream(
            model, input_path, meta, reader_cls=reader_cls,
            reader_kind=reader_kind, chunk=cfg.hann_chunk, stride=cfg.hann_stride,
            baton_path=baton_path, initial_hint=initial_hint,
            initial_hint_quality=cfg.hint_quality,
            full_hint_video_path=hint_video_path,
            full_hint_quality=cfg.hint_quality,
            rvm_path=rvm_feat_path, foundation_paths=foundation_paths,
            carry_hint_from_aux=cfg.carry_hint,
            carry_hint_quality=cfg.carry_hint_quality,
            on_baton_frames=None if cfg.low_vram else write_prediction_batches)

        if cfg.low_vram:
            _phase2_setup(model)
            baton = np.memmap(
                baton_path, dtype=np.float16, mode="r",
                shape=(T_all, model.stage1.d_16, H // 16, W // 16))
            reader = reader_cls(input_path, meta)
            try:
                for t0 in range(0, T_all, stage2_batch):
                    t1 = min(T_all, t0 + stage2_batch)
                    model_chunk, _display_chunk, _review_chunk = reader.window(t0, t1)
                    items = [
                        (t, np.asarray(baton[t]).copy(), model_chunk[t - t0])
                        for t in range(t0, t1)
                    ]
                    write_prediction_batches(items)
                    del model_chunk, _display_chunk, _review_chunk, items
            finally:
                reader.close()
                del baton
    finally:
        for writer in writers.values():
            writer.release()
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_rvm_feat.f16")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_baton.f16")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_foundation_cradio.f32")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_foundation_moge.f32")
        torch.cuda.empty_cache()
    print(f"  wrote {len(writers)} output video(s) for {clip_name} "
          f"({T_all} frames @ {W_out}x{H_out})")
