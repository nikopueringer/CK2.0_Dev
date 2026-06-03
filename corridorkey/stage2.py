"""Stage 2 — pure-transformer refinement at /16, depatchify to /1.

Design:
    • Patchify RGB to 768d at /16.
    • Concatenate stage-1 baton (d=512) with the 768d RGB patch stream:
      features = [baton | rgb_patchify].
    • 8 × mixed FullSpatial/Swin/CSwinBlock2D spatial attention (d=1280):
      full spatial at the first and middle refresh points, plus Swin/CSwin
      local/axial refinement across the four half-window shift phases.
    • In the main training phase, a SECOND zero-inited 768d patchify_2 value
      branch of raw RGB is added one block before the end. A live full-rank
      gate reads the mixed token stream and modulates that zero-init value,
      so the branch is output-neutral at insertion while still becoming
      context-dependent once it starts learning. MAE pretraining omits this
      module; it is introduced zero-init when switching to the main phase.
    • 2 more SwinBlock2D blocks.
    • Depatchify 16× (Linear(d → 4·16²) + PixelShuffle(16)); FG rows are
      initialized as the inverse of the RGB patch basis when the exact 768d
      RGB basis is available, and alpha rows start at zero.

No temporal processing here — stage 1 has done all temporal work. Stage 2
runs per-frame (B·T batched), which also means windowed spatial attention
is the tightest-possible attention shape for a 16K-token-per-frame budget
(256 windows × 64 tokens squared = 1M entries per head per layer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    PatchifyConv, Depatchify, FullSpatialBlock2D, SwinBlock2D, CSwinBlock2D,
    LayerNorm2d, icnr_init_, init_orthogonal_rgb_passthrough_,
)
from .rope import RoPE2D


def _frontloaded_qkv_kernels(total: int, kernel: int,
                            fraction: float = 1.0) -> list[int]:
    kernel = int(kernel or 0)
    total = int(total)
    if total <= 0 or kernel <= 0:
        return [0 for _ in range(max(total, 0))]
    frac = max(0.0, min(float(fraction), 1.0))
    active = total if frac >= 1.0 else int(round(total * frac))
    active = max(0, min(total, active))
    return [kernel if i < active else 0 for i in range(total)]


def _stage2_qkv_kernels(total: int, kernel: int,
                        fraction: float = 1.0) -> list[int]:
    kernel = int(kernel or 0)
    total = int(total)
    if total <= 0 or kernel <= 0:
        return [0 for _ in range(max(total, 0))]
    frac = max(0.0, min(float(fraction), 1.0))
    if frac <= 0.0:
        return [0 for _ in range(total)]

    if total != 10:
        return _frontloaded_qkv_kernels(total, kernel, frac)

    # Default v3.2 stage-2 layout with full-spatial refresh blocks:
    #   0 Full, 1 Swin(h,h), 2 CSwin, 3 Swin(h,0),
    #   4 Full, 5 CSwin, 6 Swin(0,0), 7 Swin(h,h),
    #   8 Swin(h,0), 9 Swin(0,h)
    # Prefer QKV-DW on Swin blocks, one pass per shift phase first.
    if abs(frac - 0.25) < 1e-12:
        active = 4
    else:
        active = total if frac >= 1.0 else int(round(total * frac))
    active = max(0, min(total, active))

    priority = [1, 3, 6, 9, 7, 8, 0, 4, 2, 5]
    plan = [0 for _ in range(total)]
    for i in priority[:active]:
        plan[i] = kernel
    return plan


def _qkv_kernel_at(kernels, i: int) -> int:
    if isinstance(kernels, (list, tuple)):
        return int(kernels[i]) if i < len(kernels) else 0
    return int(kernels or 0)


def _stage2_pre_blocks(d, num_heads, window_size, mlp_mult, rope,
                       num_pre_blocks=8,
                       local_mix_kernel=0,
                       local_mix_init_std=None,
                       mlp_dw_kernel=0,
                       qkv_dw_kernel=0):
    """Build the pre-patch2 pattern.

    v3.1 used 6 blocks: 2 dense → 2 CSwin → 2 dense with orthogonal shifts.
    v3.2 uses 8 blocks: 2 dense → CSwin → 2 inserted dense → CSwin → 2 dense.

    The CSwin pair in the middle gives two full-global sweeps (each block
    mixes both H and W axes via parallel head groups), with a local Swin
    refinement pair between those sweeps. Dense Swin shifts are arranged so
    the 8 pre/final dense blocks cover the four half-window phases evenly.
    """
    w = window_size
    hw = w // 2
    if num_pre_blocks == 6:
        specs = [
            ("swin", (0, 0)),
            ("swin", (hw, hw)),
            ("cswin", None),
            ("cswin", None),
            ("swin", (hw, 0)),
            ("swin", (0, hw)),
        ]
    elif num_pre_blocks == 8:
        specs = [
            ("full", None),
            ("swin", (hw, hw)),
            ("cswin", None),
            ("swin", (hw, 0)),
            ("full", None),
            ("cswin", None),
            ("swin", (0, 0)),
            ("swin", (hw, hw)),
        ]
    else:
        raise ValueError(
            f"num_pre_blocks must be 6 or 8 for the fixed CSwin pattern; got {num_pre_blocks}")
    blocks = []
    for i, (kind, shift) in enumerate(specs):
        k = _qkv_kernel_at(qkv_dw_kernel, i)
        if kind == "full":
            blocks.append(
                FullSpatialBlock2D(d, num_heads,
                                   mlp_mult=mlp_mult, rope=rope,
                                   local_mix_kernel=local_mix_kernel,
                                   local_mix_init_std=local_mix_init_std,
                                   mlp_dw_kernel=mlp_dw_kernel,
                                   qkv_dw_kernel=k)
            )
        elif kind == "swin":
            blocks.append(
                SwinBlock2D(d, num_heads, w, shift=shift,
                            mlp_mult=mlp_mult, rope=rope,
                            local_mix_kernel=local_mix_kernel,
                            local_mix_init_std=local_mix_init_std,
                            mlp_dw_kernel=mlp_dw_kernel,
                            qkv_dw_kernel=k)
            )
        else:
            blocks.append(
                CSwinBlock2D(d, num_heads, mlp_mult=mlp_mult, rope=rope,
                             local_mix_kernel=local_mix_kernel,
                             local_mix_init_std=local_mix_init_std,
                             mlp_dw_kernel=mlp_dw_kernel,
                             qkv_dw_kernel=k)
            )
    return blocks


def _stage2_final_blocks(d, num_heads, window_size, mlp_mult, rope,
                         local_mix_kernel=0,
                         local_mix_init_std=None,
                         mlp_dw_kernel=0,
                         qkv_dw_kernel=0):
    """Build the 2-block post-patch2 pattern: dense (w/2,0) → dense (0,w/2).
    Runs AFTER the late zero-init RGB pixel-skip (patchify_2) is added, so
    fresh pixel detail immediately crosses both seam axes."""
    w = window_size
    hw = w // 2
    return [
        SwinBlock2D(d, num_heads, w, shift=(hw, 0),
                    mlp_mult=mlp_mult, rope=rope,
                    local_mix_kernel=local_mix_kernel,
                    local_mix_init_std=local_mix_init_std,
                    mlp_dw_kernel=mlp_dw_kernel,
                    qkv_dw_kernel=_qkv_kernel_at(qkv_dw_kernel, 0)),
        SwinBlock2D(d, num_heads, w, shift=(0, hw),
                    mlp_mult=mlp_mult, rope=rope,
                    local_mix_kernel=local_mix_kernel,
                    local_mix_init_std=local_mix_init_std,
                    mlp_dw_kernel=mlp_dw_kernel,
                    qkv_dw_kernel=_qkv_kernel_at(qkv_dw_kernel, 1)),
    ]


class Refine4ConvNeXtBlock(nn.Module):
    """Small ConvNeXt-style block over /4 feature maps.

    NCHW is kept throughout so cuDNN sees ordinary conv tensors. The branch is
    residual, but not zero-initialized; the enclosing residual-output projection
    is zero-init, so enabling the head is still output-neutral at insertion.
    """

    def __init__(self, width: int, kernel_size: int = 7,
                 expansion: float = 2.0):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError(
                f"kernel_size must be a positive odd integer; got {kernel_size}")
        hidden = int(round(width * float(expansion)))
        self.pad = kernel_size // 2
        self.dw = nn.Conv2d(width, width, kernel_size=kernel_size,
                            padding=0, groups=width, bias=True)
        self.norm = LayerNorm2d(width)
        self.pw1 = nn.Conv2d(width, hidden, kernel_size=1, bias=True)
        self.pw2 = nn.Conv2d(hidden, width, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.pad:
            mode = 'reflect' if x.shape[-2] > self.pad and x.shape[-1] > self.pad else 'replicate'
            y = F.pad(y, (self.pad, self.pad, self.pad, self.pad), mode=mode)
        y = self.dw(y)
        y = self.norm(y)
        y = self.pw2(F.gelu(self.pw1(y), approximate='tanh'))
        return x + y


class Stage2Refine4Head(nn.Module):
    """Optional /4 conv refinement head for the final stage-2 output.

    Inputs:
      * mixed /16 stage-2 tokens after the final transformer block;
      * exact RGB pixel-unshuffle to /4 (3 * 4 * 4 = 48 channels).

    Output is a full-res residual over the normal depatchified stage-2 logits.
    The residual projection is zero-initialized, so the head is an exact no-op
    until it learns.
    """

    def __init__(
        self,
        token_dim: int,
        token_channels: int = 48,
        width: int = 96,
        blocks: int = 2,
        kernel_size: int = 7,
        expansion: float = 2.0,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        if token_channels <= 0:
            raise ValueError("token_channels must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if blocks <= 0:
            raise ValueError("blocks must be positive")
        self.token_channels = int(token_channels)
        self.rgb_channels = 3 * 4 * 4
        self.width = int(width)
        self.gradient_checkpoint = bool(gradient_checkpoint)

        self.token_up = nn.Conv2d(
            token_dim, self.token_channels * 4 * 4,
            kernel_size=1, bias=True,
        )
        icnr_init_(self.token_up, upscale_factor=4)

        in_channels = self.token_channels + self.rgb_channels
        self.in_proj = (
            nn.Identity() if in_channels == self.width
            else nn.Conv2d(in_channels, self.width, kernel_size=1, bias=True)
        )
        self.blocks = nn.ModuleList([
            Refine4ConvNeXtBlock(
                self.width, kernel_size=kernel_size, expansion=expansion)
            for _ in range(int(blocks))
        ])
        self.out = nn.Conv2d(self.width, 4 * 4 * 4, kernel_size=1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, rgb_flat: torch.Tensor,
                feat_ch_first: torch.Tensor) -> torch.Tensor:
        token = F.pixel_shuffle(self.token_up(feat_ch_first), 4)
        rgb = F.pixel_unshuffle(rgb_flat, 4)
        x = torch.cat([token, rgb], dim=1)
        x = self.in_proj(x)
        for block in self.blocks:
            if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False)
            else:
                x = block(x)
        return F.pixel_shuffle(self.out(x), 4)


class Stage2(nn.Module):
    def __init__(
        self,
        d: int = 1280,
        d_baton: int = 512,
        num_heads: int = 16,       # 1280 / 16 = 80 (head_dim)
        num_blocks: int = 10,
        num_final_blocks: int = 2,
        window_size: int = 16,
        mlp_mult: float = 2.5,
        # Per-block gradient checkpointing during training — each Swin-2D
        # block individually wrapped so only one block's activations are
        # resident at a time during backward.
        gradient_checkpoint: bool = False,
        # Overlap patchify/depatchify (see blocks.PatchifyConv / Depatchify).
        # overlap_patchify is a legacy bool alias (True → mult=3).
        # overlap_mult: 1 (no overlap) | 2 (half) | 3 (full, 3×3 neighborhood).
        overlap_patchify: bool = False,
        overlap_mult: int = 2,
        depatch_fourier_features: int = 0,
        depatch_fourier_kernel: int = 3,
        # MAE-style pretraining hooks. Disabled in the normal v3 path.
        enable_mae_mask_tokens: bool = False,
        # Optional depthwise local mixer inserted after attention and before
        # MLP in every stage-2 Swin/CSwin block. kernel=0 disables.
        local_mix_kernel: int = 0,
        local_mix_init_std: float | None = None,
        mlp_dw_kernel: int = 0,
        qkv_dw_kernel: int = 0,
        qkv_dw_fraction: float = 1.0,
        refine4_blocks: int = 0,
        refine4_width: int = 96,
        refine4_token_channels: int = 48,
        refine4_kernel: int = 7,
        refine4_expansion: float = 2.0,
        # RoPE augmentations (DINOv3-style) — see rope.py.
        rope_shift_coords: float = None,
        rope_jitter_coords: float = None,
        rope_rescale_coords: float = None,
        rope_base: float = 100.0,
    ):
        super().__init__()
        assert num_blocks >= 2, (
            f"num_blocks={num_blocks} too few — need ≥2 so the zero-init "
            f"pixel-skip can fire at least one block before the end.")
        assert 1 <= num_final_blocks < num_blocks, (
            f"num_final_blocks={num_final_blocks} must be in [1, num_blocks-1]; "
            f"num_blocks={num_blocks}")
        assert 0 < d_baton < d, (
            f"d_baton={d_baton} must be in (0, d={d}) exclusive")
        self.d = d
        self.d_baton = d_baton
        self.d_rgb = d - d_baton
        self.num_blocks = num_blocks
        self.num_final_blocks = num_final_blocks
        self.num_pre_blocks = num_blocks - num_final_blocks
        self.window_size = window_size
        self.gradient_checkpoint = gradient_checkpoint
        self.enable_mae_mask_tokens = bool(enable_mae_mask_tokens)
        self.use_patch_2 = not self.enable_mae_mask_tokens
        self.refine4_enabled = int(refine4_blocks) > 0
        qkv_plan = _stage2_qkv_kernels(
            num_blocks,
            qkv_dw_kernel,
            qkv_dw_fraction,
        )
        qkv_pre = qkv_plan[:self.num_pre_blocks]
        qkv_final = qkv_plan[self.num_pre_blocks:]
        self.qkv_dw_active_blocks = sum(1 for k in qkv_plan if k > 0)
        self.qkv_dw_total_blocks = len(qkv_plan)

        # Patchify_1: 3 → d_rgb at /16, concatenated after the baton.
        self.patch_1 = PatchifyConv(3, self.d_rgb, patch=16,
                                    zero_init_first=0,
                                    overlap=overlap_patchify,
                                    overlap_mult=overlap_mult)

        # Patchify_2 (late pixel skip): main-training only. Pretraining
        # omits it so full-RGB reconstruction cannot lean on a second raw-RGB
        # tokenization path. In phase 2, the value branch starts at zero and
        # a live gate reads the already-mixed token stream directly so token
        # magnitude remains available as a precision/confidence signal.
        if self.use_patch_2:
            self.patch_2 = PatchifyConv(3, self.d_rgb, patch=16,
                                        zero_init_all=True,
                                        overlap=overlap_patchify,
                                        overlap_mult=overlap_mult)
            self.patch_2_gate = nn.Linear(d, self.d_rgb, bias=True)
            nn.init.xavier_uniform_(self.patch_2_gate.weight, gain=0.25)
            nn.init.zeros_(self.patch_2_gate.bias)

        # Shared RoPE across Swin blocks.
        head_dim = d // num_heads
        self.rope = RoPE2D(head_dim,
                           base=rope_base,
                           shift_coords=rope_shift_coords,
                           jitter_coords=rope_jitter_coords,
                           rescale_coords=rope_rescale_coords)

        # Block layout:
        #   pre_blocks (8): full-spatial, dense (w/2,w/2), CSwin,
        #                   inserted dense (w/2,0), full-spatial,
        #                   CSwin, dense (0,0), dense (w/2,w/2)
        #   → patchify_2 zero-init RGB skip added here
        #   final_blocks (2): dense (w/2,0), dense (0,w/2)
        assert self.num_pre_blocks in (6, 8), (
            f"num_pre_blocks must be 6 or 8 for the CSwin-in-the-middle pattern; "
            f"got {self.num_pre_blocks} (num_blocks={num_blocks}, "
            f"num_final_blocks={num_final_blocks})")
        assert num_final_blocks == 2, (
            f"num_final_blocks must be 2 for the 2-phase final stack; "
            f"got {num_final_blocks}")
        self.pre_blocks = nn.ModuleList(
            _stage2_pre_blocks(d, num_heads, window_size, mlp_mult, self.rope,
                               num_pre_blocks=self.num_pre_blocks,
                               local_mix_kernel=local_mix_kernel,
                               local_mix_init_std=local_mix_init_std,
                               mlp_dw_kernel=mlp_dw_kernel,
                               qkv_dw_kernel=qkv_pre)
        )
        self.final_blocks = nn.ModuleList(
            _stage2_final_blocks(d, num_heads, window_size, mlp_mult, self.rope,
                                 local_mix_kernel=local_mix_kernel,
                                 local_mix_init_std=local_mix_init_std,
                                 mlp_dw_kernel=mlp_dw_kernel,
                                 qkv_dw_kernel=qkv_final)
        )

        # Depatchify: d → 4 alpha/RGB channels at /1 via PixelShuffle(16).
        # When dims are the production 768d RGB patch basis, FG rows invert
        # patch_1 exactly and alpha/baton rows do not affect output at init.
        self.depatchify = Depatchify(d, out_channels=4, r=16,
                                     zero_init_all=True, bias=True,
                                     overlap=overlap_patchify,
                                     overlap_mult=overlap_mult,
                                     fourier_features=depatch_fourier_features,
                                     fourier_kernel=depatch_fourier_kernel)
        self.rgb_passthrough_init = init_orthogonal_rgb_passthrough_(
            self.patch_1, self.depatchify,
            token_offset=self.d_baton,
            output_rgb_offset=1,
            downsample=1,
        )
        self.refine4 = (
            Stage2Refine4Head(
                d, token_channels=refine4_token_channels,
                width=refine4_width, blocks=int(refine4_blocks),
                kernel_size=refine4_kernel, expansion=refine4_expansion,
                gradient_checkpoint=gradient_checkpoint,
            )
            if self.refine4_enabled else None
        )
        if self.enable_mae_mask_tokens:
            self.mae_rgb_token = nn.Parameter(torch.zeros(self.d_rgb))

    # -----------------------------------------------------------------

    def _apply_mae_rgb_token(
        self,
        x: torch.Tensor,
        mae_mask_16: torch.Tensor | None,
        B: int,
        T: int,
        name: str,
    ) -> torch.Tensor:
        if mae_mask_16 is None:
            return x
        if not self.enable_mae_mask_tokens:
            raise RuntimeError(f"{name} received an MAE mask but MAE tokens are disabled")
        BT, C, H16, W16 = x.shape
        if C != self.d_rgb:
            raise ValueError(f"{name} channel count {C} != d_rgb {self.d_rgb}")
        expected = (B, T, 1, H16, W16)
        if tuple(mae_mask_16.shape) != expected:
            raise ValueError(
                f"{name} mask shape {tuple(mae_mask_16.shape)} != expected {expected}")
        mask = mae_mask_16.reshape(BT, 1, H16, W16).to(device=x.device,
                                                       dtype=torch.bool)
        token = self.mae_rgb_token.to(device=x.device, dtype=x.dtype).view(1, C, 1, 1)
        return torch.where(mask, token, x)

    def forward(
        self,
        rgb: torch.Tensor,     # (B, T, 3, H, W) signed asinh encoded scene-linear RGB
        baton: torch.Tensor,   # (B, T, d_baton, H/16, W/16) from stage 1
        mae_mask_16: 'torch.Tensor | None' = None,  # (B,T,1,H/16,W/16)
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Returns logits (B, T, 4, H, W), or (refined, base) when refine4 is on."""
        B, T, _, H, W = rgb.shape
        assert H % 16 == 0 and W % 16 == 0, (
            f"Stage2: H,W must be /16-aligned; got {(H, W)}")
        assert baton.shape == (B, T, self.d_baton, H // 16, W // 16), (
            f"baton shape mismatch: got {baton.shape}, "
            f"expected {(B, T, self.d_baton, H // 16, W // 16)}")

        BT = B * T
        rgb_flat = rgb.reshape(BT, 3, H, W)

        # Patchify_1 (RGB → d_rgb at /16), then concatenate after the baton.
        rgb_feat = self.patch_1(rgb_flat)                     # (BT, d_rgb, H/16, W/16)
        rgb_feat = self._apply_mae_rgb_token(
            rgb_feat, mae_mask_16, B, T, "stage2 patch_1")

        baton_flat = baton.reshape(BT, self.d_baton, H // 16, W // 16)
        feat = torch.cat([baton_flat, rgb_feat], dim=1)       # (BT, d, H/16, W/16)

        # Swin-2D permute to (BT, H/16, W/16, d)
        x = feat.permute(0, 2, 3, 1).contiguous()

        # Per-block gradient checkpointing: only one block's activations
        # are resident at a time during backward when gradient_checkpoint is on.
        def run_block(blk, x_in):
            if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(
                    blk, x_in, use_reentrant=False,
                )
            return blk(x_in)

        # Pre-patch_2 blocks (num_pre_blocks)
        for blk in self.pre_blocks:
            x = run_block(blk, x)

        if self.use_patch_2:
            # Late pixel skip: zero-init patchify_2 of raw RGB, modulated by
            # a full-rank gate over the mixed token stream. The value branch
            # is exactly zero at insertion, so the live zero-bias gate only
            # affects gradients until patch_2 wakes up.
            gate = F.silu(self.patch_2_gate(x))                # (BT, H/16, W/16, d_rgb)
            patch2 = self.patch_2(rgb_flat)                    # (BT, d_rgb, H/16, W/16)
            patch2 = patch2.permute(0, 2, 3, 1).contiguous() * gate
            x = x + F.pad(patch2, (self.d_baton, 0))

        # Post-patch_2 final blocks (num_final_blocks)
        for blk in self.final_blocks:
            x = run_block(blk, x)

        # Depatchify to /1. With production dims and non-overlap patching,
        # FG rows are the exact inverse of patch_1 at init; alpha rows and
        # baton columns are zero.
        feat_ch_first = x.permute(0, 3, 1, 2).contiguous()     # (BT, d, H/16, W/16)
        out_raw = self.depatchify(feat_ch_first)               # (BT, 4, H, W)
        if self.refine4 is not None:
            refined = out_raw + self.refine4(rgb_flat, feat_ch_first)
            return refined.reshape(B, T, 4, H, W), out_raw.reshape(B, T, 4, H, W)
        return out_raw.reshape(B, T, 4, H, W)
