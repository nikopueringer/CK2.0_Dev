"""Top-level model assembly for CorridorKey v3.

Forward contract:
    rgb:       (B, T, 3, H, W)          signed asinh encoded scene-linear RGB
    hint:      (B, T, 1, H, W)          in [0, 1]
    rgb_cradio:(B, T, 3, H/4,  W/4)     for C-RADIO/RVM (patch 16 → /64 grid)
    rgb_moge:  (B, T, 3, 7H/32, 7W/32)  for MoGe (patch 14 → /64 grid)
    t_stride:  float — temporal RoPE scaling factor

H and W must be multiples of 64 (shape_quantum=64). The dataset
guarantees this.

Output dict:
    'alpha':       (B, T, 1, H, W) — stage-2 refined
    'fg':          (B, T, 3, H, W) asinh encoded — stage-2 refined
    'alpha_aux':   (B, T, 1, H/2, W/2) — stage-1 auxiliary prediction
    'fg_aux':      (B, T, 3, H/2, W/2) — stage-1 auxiliary prediction
    'alpha_base':  optional (B, T, 1, H, W) — pre-/4-refine stage-2 output
    'fg_base':     optional (B, T, 3, H, W) — pre-/4-refine stage-2 output
    'alpha_bottleneck': optional (B, T, 1, H/8, W/8) — /64 supervision head
    'fg_bottleneck':    optional (B, T, 3, H/8, W/8) — /64 supervision head
    'baton':       (B, T, d_baton, H/16, W/16) — debug/intermediate
"""

import torch
import torch.nn as nn

from .foundations import FrozenFoundations
from .stage1 import Stage1
from .stage2 import Stage2
from .rope import refresh_all_augs


class CorridorKeyV3(nn.Module):
    """Transformer-native two-stage video matting model.

    Stage 1 produces a /16 baton (d=512) plus an auxiliary /2 prediction
    for deep supervision. Stage 2 refines the baton + raw RGB to a /1
    alpha+FG output via pure-transformer Swin-2D + depatchify.

    Temporal mixing lives primarily in stage 1: full spatiotemporal attention
    at /64, Swin-3D at /16, and optional /64 recurrent memory sites for
    streaming identity/decision carry.
    """

    def __init__(
        self,
        # Stage 1
        d_16: int = 512,
        d_64: int = 1536,
        n_enc_16: int = 2,
        n_body_64: int = 10,
        n_dec_16: int = 6,
        num_heads_16: int = 8,
        num_heads_64: int = 16,
        window_HW: int = 16,
        stage1_local_mix_kernel: int = 7,
        stage1_local_mix_init_std: float | None = None,
        stage1_body64_local_mix_kernel: int = 7,
        stage1_body64_local_mix_init_std: float | None = None,
        stage1_body64_mlp_hidden: int | None = None,
        stage1_body64_direct_concat: bool = False,
        stage1_recurrent_64_sites: tuple[int, ...] | None = None,
        stage1_hint_adaln_blocks: int = 2,
        stage1_mid_16_blocks: int = 4,
        # Stage 2
        d_stage2: int = 1280,
        n_blocks_stage2: int = 10,
        n_final_blocks_stage2: int = 2,
        num_heads_stage2: int = 16,
        window_size_stage2: int = 16,
        stage2_local_mix_kernel: int = 7,
        stage2_local_mix_init_std: float | None = None,
        refine4_blocks: int = 0,
        refine4_width: int = 96,
        refine4_token_channels: int = 48,
        refine4_kernel: int = 7,
        refine4_expansion: float = 2.0,
        # Shared
        mlp_mult: float = 2.5,
        mlp_dw_kernel: int = 0,
        qkv_dw_kernel: int = 7,
        qkv_dw_fraction: float = 0.25,
        # Foundations
        use_moge: bool = True,
        use_cradio: bool = False,
        cradio_repo: str = 'nvidia/C-RADIOv4-SO400M',
        moge_checkpoint: str = 'Ruicheng/moge-2-vitb-normal',
        # RVM is enabled by passing a ckpt path. RVM has patch 16, so
        # we feed it the same H/4-sized input C-RADIO sees; features land at
        # the /64 grid naturally, no resize or aspect-distortion needed.
        rvm_variant: str = 'small',
        rvm_ckpt: str = None,
        foundation_stats_path: str = None,
        device: str = 'cuda',
        # Training
        gradient_checkpoint: bool = True,
        # Architecture ablation knob: overlap patchify/depatchify.
        # overlap_patchify is a legacy bool alias (True → mult=3).
        # overlap_mult: 1 (no overlap), 2 (half), 3 (full 3×3 neighborhood).
        overlap_patchify: bool = False,
        overlap_mult: int = 2,
        depatch_fourier_features: int = 0,
        depatch_fourier_kernel: int = 3,
        # MAE-style pretraining hooks plus the /64 bottleneck supervision head.
        # The bottleneck head default matches phase-2 training.
        enable_mae_mask_tokens: bool = False,
        bottleneck_head_stride: int | None = 8,
        # DINOv3-style RoPE augmentations (training-time, per-forward coherent)
        rope_shift_coords: float = None,
        rope_jitter_coords: float = None,
        rope_rescale_coords: float = None,
        rope_base: float = 100.0,
    ):
        super().__init__()
        self.gradient_checkpoint = gradient_checkpoint
        self.d_16 = d_16

        self.foundations = FrozenFoundations(
            device=device,
            use_moge=use_moge,
            use_cradio=use_cradio, cradio_repo=cradio_repo,
            moge_checkpoint=moge_checkpoint,
            rvm_variant=rvm_variant,
            rvm_ckpt=rvm_ckpt,
            stats_path=foundation_stats_path,
        )

        self.stage1 = Stage1(
            d_16=d_16, d_64=d_64,
            cradio_dim=self.foundations.cradio_dim,
            moge_dim=self.foundations.moge_dim,
            rvm_dim=self.foundations.rvm_dim,
            n_enc_16=n_enc_16, n_body_64=n_body_64, n_dec_16=n_dec_16,
            num_heads_16=num_heads_16, num_heads_64=num_heads_64,
            window_HW=window_HW, mlp_mult=mlp_mult,
            local_mix_kernel=stage1_local_mix_kernel,
            local_mix_init_std=stage1_local_mix_init_std,
            body64_local_mix_kernel=stage1_body64_local_mix_kernel,
            body64_local_mix_init_std=stage1_body64_local_mix_init_std,
            body64_mlp_hidden=stage1_body64_mlp_hidden,
            body64_direct_concat=stage1_body64_direct_concat,
            recurrent_64_sites=stage1_recurrent_64_sites,
            hint_adaln_blocks=stage1_hint_adaln_blocks,
            mid_16_blocks=stage1_mid_16_blocks,
            mlp_dw_kernel=mlp_dw_kernel,
            qkv_dw_kernel=qkv_dw_kernel,
            qkv_dw_fraction=qkv_dw_fraction,
            gradient_checkpoint=gradient_checkpoint,
            overlap_patchify=overlap_patchify,
            overlap_mult=overlap_mult,
            depatch_fourier_features=depatch_fourier_features,
            depatch_fourier_kernel=depatch_fourier_kernel,
            enable_mae_mask_tokens=enable_mae_mask_tokens,
            bottleneck_head_stride=bottleneck_head_stride,
            rope_shift_coords=rope_shift_coords,
            rope_jitter_coords=rope_jitter_coords,
            rope_rescale_coords=rope_rescale_coords,
            rope_base=rope_base,
        )

        self.stage2 = Stage2(
            d=d_stage2, d_baton=d_16,
            num_heads=num_heads_stage2,
            num_blocks=n_blocks_stage2,
            num_final_blocks=n_final_blocks_stage2,
            window_size=window_size_stage2,
            mlp_mult=mlp_mult,
            gradient_checkpoint=gradient_checkpoint,
            overlap_patchify=overlap_patchify,
            overlap_mult=overlap_mult,
            depatch_fourier_features=depatch_fourier_features,
            depatch_fourier_kernel=depatch_fourier_kernel,
            enable_mae_mask_tokens=enable_mae_mask_tokens,
            local_mix_kernel=stage2_local_mix_kernel,
            local_mix_init_std=stage2_local_mix_init_std,
            mlp_dw_kernel=mlp_dw_kernel,
            qkv_dw_kernel=qkv_dw_kernel,
            qkv_dw_fraction=qkv_dw_fraction,
            refine4_blocks=refine4_blocks,
            refine4_width=refine4_width,
            refine4_token_channels=refine4_token_channels,
            refine4_kernel=refine4_kernel,
            refine4_expansion=refine4_expansion,
            rope_shift_coords=rope_shift_coords,
            rope_jitter_coords=rope_jitter_coords,
            rope_rescale_coords=rope_rescale_coords,
            rope_base=rope_base,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        hint: torch.Tensor,
        rgb_cradio: 'torch.Tensor | None',
        rgb_moge: torch.Tensor,
        rgb_rvm: 'torch.Tensor | None' = None,
        rvm_state: 'torch.Tensor | None' = None,
        t_stride: float = 1.0,
        mae_mask_16: 'torch.Tensor | None' = None,
        mae_mask_64: 'torch.Tensor | None' = None,
        mae_vfm_mask_64: 'torch.Tensor | None' = None,
        hint_quality: 'torch.Tensor | None' = None,
        stage1_recurrent_states: 'tuple[torch.Tensor | None, ...] | None' = None,
    ) -> dict:
        B, T, _, H, W = rgb.shape
        assert H % 64 == 0 and W % 64 == 0, (
            f"H, W must be multiples of 64 (shape_quantum); got {(H, W)}.")
        if (mae_mask_16 is None) != (mae_mask_64 is None):
            raise ValueError("mae_mask_16 and mae_mask_64 must be passed together")
        if mae_vfm_mask_64 is not None and mae_mask_64 is None:
            raise ValueError("mae_vfm_mask_64 requires mae_mask_64")

        # Refresh RoPE augmentations ONCE per forward so every RoPE call
        # inside stage1/stage2 (across all blocks) uses the same sample.
        # In eval mode this clears the state to identity (deterministic).
        refresh_all_augs(self, device=rgb.device)

        # Foundations — per-frame, no gradients.
        BT = B * T
        rgb_moge_flat = rgb_moge.reshape(BT, 3, *rgb_moge.shape[-2:])
        rgb_cradio_flat = (
            rgb_cradio.reshape(BT, 3, *rgb_cradio.shape[-2:])
            if rgb_cradio is not None else None
        )
        # RVM input: feed the same H/4-sized tensor C-RADIO sees. RVM has
        # patch 16, so the patch grid lands on /64 directly —
        # no resize, no aspect distortion. If caller supplied a custom
        # rgb_rvm explicitly, honor that instead.
        if self.foundations.rvm is not None and rgb_rvm is None:
            assert rgb_cradio is not None, (
                "RVM inline extraction requires an H/4 C-RADIO/RVM input")
            H_e, W_e = rgb_cradio.shape[-2:]
            rgb_rvm = rgb_cradio.reshape(B, T, 3, H_e, W_e)
        with torch.no_grad():
            foundation_feats = self.foundations(
                rgb_moge=rgb_moge_flat,
                rgb_cradio=rgb_cradio_flat,
                rgb_rvm=rgb_rvm, rvm_state=rvm_state,
            )
        cradio_feat = foundation_feats.get('cradio')
        moge_feat = foundation_feats.get('moge')
        rvm_feat = foundation_feats.get('rvm')
        rvm_state_out = foundation_feats.get('rvm_state')
        if cradio_feat is not None:
            cradio_feat = cradio_feat.reshape(B, T, -1, H // 64, W // 64)
        else:
            cradio_feat = rgb.new_zeros(B, T, 0, H // 64, W // 64)
        if moge_feat is not None:
            moge_feat = moge_feat.reshape(B, T, -1, H // 64, W // 64)
        else:
            moge_feat = rgb.new_zeros(B, T, 0, H // 64, W // 64)
        if rvm_feat is not None:
            rvm_feat = rvm_feat.reshape(B, T, rvm_feat.shape[-3],
                                         rvm_feat.shape[-2], rvm_feat.shape[-1])

        # Stage 1 & 2 — per-block gradient checkpointing is done INSIDE each
        # stage's forward (see Stage1._ckpt_block, Stage2.run_block). No outer
        # stage-level wrap — per-block peak activations are much smaller than
        # stage-level, and double-wrapping would just cause nested recompute.
        stage1_out = self.stage1(
            rgb, hint, moge_feat,
            cradio_feat=cradio_feat,
            rvm_feat=rvm_feat, t_stride=t_stride,
            mae_mask_16=mae_mask_16, mae_mask_64=mae_mask_64,
            mae_vfm_mask_64=mae_vfm_mask_64,
            hint_quality=hint_quality,
            recurrent_states=stage1_recurrent_states,
        )
        baton = stage1_out['baton']           # (B, T, d_16, H/16, W/16)
        aux = stage1_out['aux']               # (B, T, 4, H/2, W/2)
        bottleneck = stage1_out.get('bottleneck')
        recurrent_states_out = stage1_out.get('recurrent_states')
        del stage1_out, foundation_feats
        del cradio_feat, moge_feat, rvm_feat
        del rgb_moge_flat, rgb_cradio_flat, rgb_rvm

        logits_out = self.stage2(rgb, baton, mae_mask_16=mae_mask_16)
        if isinstance(logits_out, tuple):
            logits, logits_base = logits_out
        else:
            logits, logits_base = logits_out, None

        # Split outputs
        alpha = logits[:, :, 0:1]
        fg = logits[:, :, 1:4]
        alpha_aux = aux[:, :, 0:1]
        fg_aux = aux[:, :, 1:4]

        out = {
            'alpha': alpha,
            'fg': fg,
            'alpha_aux': alpha_aux,
            'fg_aux': fg_aux,
            'baton': baton,
        }
        if logits_base is not None:
            out['alpha_base'] = logits_base[:, :, 0:1]
            out['fg_base'] = logits_base[:, :, 1:4]
        if bottleneck is not None:
            out['alpha_bottleneck'] = bottleneck[:, :, 0:1]
            out['fg_bottleneck'] = bottleneck[:, :, 1:4]
        if recurrent_states_out is not None:
            out['stage1_recurrent_states'] = recurrent_states_out
        if rvm_state_out is not None:
            out['rvm_state'] = rvm_state_out
        return out
