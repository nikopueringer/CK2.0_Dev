"""Rotary Position Embeddings — 2D (stage 2) and 3D (stage 1) variants.

DINOv3/EUPE-style: normalized [-1, 1] cell-center coordinates with
training-time augmentations (shift, per-axis jitter, global rescale).

CorridorKey v3 uses:
  • 3D RoPE (T, H, W) inside full-attention body at /64 and inside Swin-3D
    windows at /16 (stage 1).
  • 2D RoPE (H, W)      inside Swin-2D windows at /16 (stage 2; no temporal).

Periods (lengths in normalized-coord units):
    periods[i] = base ** (2i / (D_axis))   for i in [0, D_axis // 2)
    angles = 2π · coord / period
where `coord ∈ [-1, 1]` is the cell-center coord of a token in the
(sub-)grid and `D_axis` is the portion of head_dim assigned to this axis.

Head-dim split:
    2D:  head_dim/4 periods per axis × 2 axes → head_dim/2 cos/sin pairs
    3D:  head_dim/2 cos/sin pairs split as evenly as possible across T/H/W
(2D asserts divisibility by 4; 3D only requires an even head_dim)

Application format (kept from original):
    x[0::2] / x[1::2] interleave — cos/sin tensors are (N, head_dim//2),
    one value per pair.

Training augmentations, sampled once per forward pass via refresh_augs()
so every RoPE call during the same forward gets a CONSISTENT positional
encoding:
  • shift_coords   (s):  per-axis additive shift ~ U[-s, s]
  • jitter_coords  (J):  per-axis multiplicative log-uniform in [1/J, J]
                         (aspect-ratio BREAKING — each axis jitters
                         independently)
  • rescale_coords (R):  global multiplicative log-uniform in [1/R, R]
                         (aspect-ratio PRESERVING — same scale on every
                         axis)
All disabled at eval.

Convention:
  • Token ordering: (t, h, w) with w varying fastest for 3D;
                    (h, w) with w fastest for 2D.
  • All three axes use the same `base` (default 100).
  • Coord normalization per EUPE's "separate" mode: coord_a = (i+0.5)/N_a.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _compute_periods(dim: int, base: float) -> torch.Tensor:
    """Period lengths in normalized-coord units. Returns (dim//2,) tensor."""
    assert dim % 2 == 0, f"_compute_periods dim={dim} must be even"
    exps = torch.arange(0, dim // 2, dtype=torch.float32) * 2.0 / dim
    return base ** exps  # Range: [1, base^((dim-2)/dim)]


def _normalized_coords(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Cell-center normalized coords in [-1, 1] for n tokens along an axis.
    Matches DINOv3 / EUPE 'separate' normalization: (i+0.5)/n then 2x-1."""
    x = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n   # [0.5/n, (n-0.5)/n]
    return 2.0 * x - 1.0


def _apply_axis_rotary(coord: torch.Tensor, periods: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given coord (N,) in normalized units and periods (D_axis//2,),
    return (cos, sin) of shape (N, D_axis//2)."""
    angles = 2.0 * math.pi * coord.unsqueeze(1) / periods.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
                 ) -> torch.Tensor:
    """Apply rotation to the head dimension of `x`.

    Args:
        x: (..., N, head_dim) — queries or keys.
        cos/sin: (N, head_dim // 2) — frequency tables.

    Returns rotated tensor of same shape as `x`.
    """
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Augmentation state (shared logic between RoPE2D and RoPE3D)
# ---------------------------------------------------------------------------


class _RopeAugBase(nn.Module):
    """Holds training-time augmentation params and a per-forward cached
    sample. Subclasses fill in axis geometry + coord construction."""

    def __init__(self, base: float = 100.0,
                 shift_coords: Optional[float] = None,
                 jitter_coords: Optional[float] = None,
                 rescale_coords: Optional[float] = None,
                 n_axes: int = 2):
        super().__init__()
        self.base = base
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.n_axes = n_axes
        # Cached per-forward augmentation: (shift_vec, jitter_log_vec, rescale_log_scalar).
        # `None` means "use identity" (eval-time or uninitialized).
        self._aug_shift: Optional[List[float]] = None
        self._aug_jitter_log: Optional[List[float]] = None
        self._aug_rescale_log: Optional[float] = None

    def refresh_augs(self, device=None, force_eval: bool = False) -> None:
        """Call at the start of each model forward during training; clears
        the aug state (identity) during eval."""
        if force_eval or not self.training:
            self._aug_shift = None
            self._aug_jitter_log = None
            self._aug_rescale_log = None
            return
        if self.shift_coords is not None and self.shift_coords > 0:
            s = float(self.shift_coords)
            self._aug_shift = (torch.empty(self.n_axes)
                               .uniform_(-s, s).tolist())
        else:
            self._aug_shift = None
        if self.jitter_coords is not None and self.jitter_coords > 1.0:
            lg = math.log(float(self.jitter_coords))
            self._aug_jitter_log = (torch.empty(self.n_axes)
                                    .uniform_(-lg, lg).tolist())
        else:
            self._aug_jitter_log = None
        if self.rescale_coords is not None and self.rescale_coords > 1.0:
            lg = math.log(float(self.rescale_coords))
            self._aug_rescale_log = float(torch.empty(1)
                                          .uniform_(-lg, lg).item())
        else:
            self._aug_rescale_log = None

    def _aug_coord(self, coord: torch.Tensor, axis_idx: int) -> torch.Tensor:
        """Apply current-forward augs to a 1D coord tensor for axis `axis_idx`."""
        out = coord
        if self._aug_shift is not None:
            out = out + self._aug_shift[axis_idx]
        if self._aug_jitter_log is not None:
            out = out * math.exp(self._aug_jitter_log[axis_idx])
        if self._aug_rescale_log is not None:
            out = out * math.exp(self._aug_rescale_log)
        return out


# ---------------------------------------------------------------------------
# RoPE3D  — used by stage 1 (Swin-3D enc/dec at /16, Full-ST body at /64)
# ---------------------------------------------------------------------------


class RoPE3D(_RopeAugBase):
    """3D Rotary Position Embedding for spatiotemporal tokens.

    head_dim split: rotary pairs are split nearly evenly across T/H/W.
    Exact thirds are used when available; otherwise the extra pairs are given
    to H/W first, which lets power-of-two head dims such as 64 work cleanly.
    """

    def __init__(self, head_dim: int,
                 base: float = 100.0,
                 shift_coords: Optional[float] = None,
                 jitter_coords: Optional[float] = None,
                 rescale_coords: Optional[float] = None):
        super().__init__(base=base,
                         shift_coords=shift_coords,
                         jitter_coords=jitter_coords,
                         rescale_coords=rescale_coords,
                         n_axes=3)
        assert head_dim % 2 == 0, (
            f"3D RoPE head_dim={head_dim} must be even for cos/sin pairs.")
        self.head_dim = head_dim
        total_pairs = head_dim // 2
        base_pairs = total_pairs // 3
        rem = total_pairs - base_pairs * 3
        pairs_t = base_pairs
        pairs_h = base_pairs + (1 if rem >= 2 else 0)
        pairs_w = base_pairs + (1 if rem >= 1 else 0)
        self.dim_t = pairs_t * 2
        self.dim_h = pairs_h * 2
        self.dim_w = pairs_w * 2

        self.register_buffer('periods_t',
                             _compute_periods(self.dim_t, base),
                             persistent=False)
        self.register_buffer('periods_h',
                             _compute_periods(self.dim_h, base),
                             persistent=False)
        self.register_buffer('periods_w',
                             _compute_periods(self.dim_w, base),
                             persistent=False)

    def get_cos_sin(
        self,
        num_t: int, num_h: int, num_w: int,
        device: torch.device = None,
        t_stride: float = 1.0,
        # Legacy args kept for call-site compat; ignored in the normalized
        # formulation (Swin windows use local-to-window normalization).
        t_offset: int = 0, h_offset: int = 0, w_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (num_t * num_h * num_w, head_dim // 2) cos and sin tables.

        Positions are cell-center normalized to [-1, 1] per axis, optionally
        augmented by the cached refresh_augs() sample. The `t_stride` scales
        the temporal axis post-normalization (dataset-level frame-stride /
        frame-rate knob, orthogonal to the training jitter).
        """
        device = device or self.periods_t.device
        coord_t = _normalized_coords(num_t, device=device) * t_stride
        coord_h = _normalized_coords(num_h, device=device)
        coord_w = _normalized_coords(num_w, device=device)
        coord_t = self._aug_coord(coord_t, axis_idx=0)
        coord_h = self._aug_coord(coord_h, axis_idx=1)
        coord_w = self._aug_coord(coord_w, axis_idx=2)

        cos_t, sin_t = _apply_axis_rotary(coord_t, self.periods_t.to(device))
        cos_h, sin_h = _apply_axis_rotary(coord_h, self.periods_h.to(device))
        cos_w, sin_w = _apply_axis_rotary(coord_w, self.periods_w.to(device))

        N = num_t * num_h * num_w
        cos_t_exp = cos_t[:, None, None, :].expand(num_t, num_h, num_w, -1).reshape(N, -1)
        sin_t_exp = sin_t[:, None, None, :].expand(num_t, num_h, num_w, -1).reshape(N, -1)
        cos_h_exp = cos_h[None, :, None, :].expand(num_t, num_h, num_w, -1).reshape(N, -1)
        sin_h_exp = sin_h[None, :, None, :].expand(num_t, num_h, num_w, -1).reshape(N, -1)
        cos_w_exp = cos_w[None, None, :, :].expand(num_t, num_h, num_w, -1).reshape(N, -1)
        sin_w_exp = sin_w[None, None, :, :].expand(num_t, num_h, num_w, -1).reshape(N, -1)

        cos = torch.cat([cos_t_exp, cos_h_exp, cos_w_exp], dim=-1)
        sin = torch.cat([sin_t_exp, sin_h_exp, sin_w_exp], dim=-1)
        return cos, sin


# ---------------------------------------------------------------------------
# RoPE2D  — used by stage 2 (Swin-2D at /16)
# ---------------------------------------------------------------------------


class RoPE2D(_RopeAugBase):
    """2D Rotary Position Embedding for spatial-only tokens (stage 2).

    head_dim split evenly between H and W. head_dim must be divisible by 4.
    """

    def __init__(self, head_dim: int,
                 base: float = 100.0,
                 shift_coords: Optional[float] = None,
                 jitter_coords: Optional[float] = None,
                 rescale_coords: Optional[float] = None):
        super().__init__(base=base,
                         shift_coords=shift_coords,
                         jitter_coords=jitter_coords,
                         rescale_coords=rescale_coords,
                         n_axes=2)
        assert head_dim % 4 == 0, (
            f"2D RoPE head_dim={head_dim} must be divisible by 4.")
        self.head_dim = head_dim
        dim_each = head_dim // 2
        self.dim_h = dim_each
        self.dim_w = dim_each
        self.register_buffer('periods_h',
                             _compute_periods(dim_each, base),
                             persistent=False)
        self.register_buffer('periods_w',
                             _compute_periods(dim_each, base),
                             persistent=False)

    def get_cos_sin(
        self, num_h: int, num_w: int,
        device: torch.device = None,
        # Legacy kwargs kept for call-site compat; unused in normalized form.
        h_offset: int = 0, w_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (num_h * num_w, head_dim // 2) cos/sin tables."""
        device = device or self.periods_h.device
        coord_h = _normalized_coords(num_h, device=device)
        coord_w = _normalized_coords(num_w, device=device)
        coord_h = self._aug_coord(coord_h, axis_idx=0)
        coord_w = self._aug_coord(coord_w, axis_idx=1)

        cos_h, sin_h = _apply_axis_rotary(coord_h, self.periods_h.to(device))
        cos_w, sin_w = _apply_axis_rotary(coord_w, self.periods_w.to(device))
        N = num_h * num_w
        cos_h_exp = cos_h[:, None, :].expand(num_h, num_w, -1).reshape(N, -1)
        sin_h_exp = sin_h[:, None, :].expand(num_h, num_w, -1).reshape(N, -1)
        cos_w_exp = cos_w[None, :, :].expand(num_h, num_w, -1).reshape(N, -1)
        sin_w_exp = sin_w[None, :, :].expand(num_h, num_w, -1).reshape(N, -1)
        cos = torch.cat([cos_h_exp, cos_w_exp], dim=-1)
        sin = torch.cat([sin_h_exp, sin_w_exp], dim=-1)
        return cos, sin


# ---------------------------------------------------------------------------
# Convenience: refresh all RoPE modules' augs in one call
# ---------------------------------------------------------------------------


def refresh_all_augs(module: nn.Module, device=None, force_eval: bool = False) -> None:
    """Walk a module tree and call refresh_augs on every RoPE instance.

    Invoke once at the top of each model forward pass so all RoPE modules
    in the model get a COHERENT augmentation sample for this step.
    """
    for m in module.modules():
        if isinstance(m, _RopeAugBase):
            m.refresh_augs(device=device, force_eval=force_eval)
