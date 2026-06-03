#!/usr/bin/env python3
"""Export a single CorridorKey v2 runtime weight file.

This strips the training checkpoint down to one selected model state dict plus
the model configuration needed by the runtime loader.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def _as_dict(args_obj):
    if args_obj is None:
        return {}
    if isinstance(args_obj, dict):
        return dict(args_obj)
    if hasattr(args_obj, "__dict__"):
        return dict(vars(args_obj))
    return {}


def _get(src, key, default):
    return src.get(key, default) if isinstance(src, dict) else default


def _parse_ema(ckpt, source: str):
    if source == "student":
        return ckpt["state_dict"], None
    ema = ckpt.get("ema")
    if not isinstance(ema, dict):
        raise RuntimeError("source checkpoint has no EMA shadows")
    if source == "standard":
        return ema["standard"], {"kind": "standard", "beta": ema.get("standard_beta")}
    if source.startswith("cascade_a_") or source.startswith("cascade_b_"):
        parts = source.split("_")
        cascade = "_".join(parts[:2])
        level = int(parts[2])
        if cascade not in ema:
            raise RuntimeError(f"source checkpoint has no EMA cascade {cascade}")
        shadows = ema[cascade]
        if level < 0 or level >= len(shadows):
            raise RuntimeError(f"{cascade} has {len(shadows)} levels, not {level + 1}")
        return shadows[level], {
            "kind": cascade,
            "level": level,
            "beta": ema.get(f"{cascade}_beta"),
        }
    raise RuntimeError(f"unrecognized EMA source: {source}")


def _normalize_state_dict(sd):
    out = {}
    for key, value in sd.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        out[key] = value.detach().cpu()
    return out


def _model_config(ck_args):
    # Paths are package-relative and resolved by corridorkey.runtime.
    return {
        "d_16": _get(ck_args, "d_16", 512),
        "d_64": _get(ck_args, "d_64", 1536),
        "d_stage2": _get(ck_args, "d_stage2", 1280),
        "n_enc_16": _get(ck_args, "n_enc_16", 2),
        "n_body_64": _get(ck_args, "n_body_64", 10),
        "n_dec_16": _get(ck_args, "n_dec_16", 6),
        "num_heads_16": _get(ck_args, "num_heads_16", 8),
        "num_heads_64": _get(ck_args, "num_heads_64", 16),
        "num_heads_stage2": _get(ck_args, "num_heads_stage2", 16),
        "window_HW": _get(ck_args, "window_HW", 16),
        "window_size_stage2": _get(ck_args, "window_size_stage2", 16),
        "stage1_local_mix_kernel": _get(ck_args, "stage1_local_mix_kernel", 7),
        "stage1_local_mix_init_std": _get(ck_args, "stage1_local_mix_init_std", None),
        "stage1_body64_local_mix_kernel": _get(ck_args, "stage1_body64_local_mix_kernel", 7),
        "stage1_body64_local_mix_init_std": _get(ck_args, "stage1_body64_local_mix_init_std", None),
        "stage1_body64_mlp_hidden": _get(ck_args, "stage1_body64_mlp_hidden", None),
        "stage1_body64_direct_concat": _get(ck_args, "stage1_body64_direct_concat", False),
        "stage1_recurrent_64_sites": (),
        "stage1_hint_adaln_blocks": _get(ck_args, "stage1_hint_adaln_blocks", 2),
        "stage1_mid_16_blocks": _get(ck_args, "stage1_mid_16_blocks", 4),
        "n_blocks_stage2": _get(ck_args, "n_blocks_stage2", 10),
        "n_final_blocks_stage2": _get(ck_args, "n_final_blocks_stage2", 2),
        "stage2_local_mix_kernel": _get(ck_args, "stage2_local_mix_kernel", 7),
        "stage2_local_mix_init_std": _get(ck_args, "stage2_local_mix_init_std", None),
        "refine4_blocks": _get(ck_args, "refine4_blocks", 0),
        "refine4_width": _get(ck_args, "refine4_width", 96),
        "refine4_token_channels": _get(ck_args, "refine4_token_channels", 48),
        "refine4_kernel": _get(ck_args, "refine4_kernel", 7),
        "refine4_expansion": _get(ck_args, "refine4_expansion", 2.0),
        "mlp_mult": _get(ck_args, "mlp_mult", 2.5),
        "mlp_dw_kernel": _get(ck_args, "mlp_dw_kernel", 0),
        "qkv_dw_kernel": _get(ck_args, "qkv_dw_kernel", 7),
        "qkv_dw_fraction": _get(ck_args, "qkv_dw_fraction", 0.25),
        "use_moge": True,
        "use_cradio": True,
        "cradio_repo": "vfms/cradio/C-RADIOv4-SO400M",
        "moge_checkpoint": "vfms/moge/model.pt",
        "rvm_variant": _get(ck_args, "rvm_variant", "small"),
        "rvm_ckpt": "vfms/rvm/pretrain_rvm_small16_256_204031069.npz",
        "gradient_checkpoint": False,
        "overlap_patchify": _get(ck_args, "overlap_patchify", False),
        "overlap_mult": _get(ck_args, "overlap_mult", 2),
        "depatch_fourier_features": _get(ck_args, "depatch_fourier_features", 0),
        "depatch_fourier_kernel": _get(ck_args, "depatch_fourier_kernel", 3),
        "enable_mae_mask_tokens": bool(_get(ck_args, "enable_mae_mask_tokens", False)),
        "bottleneck_head_stride": _get(ck_args, "bottleneck_head_stride", 8),
        "rope_shift_coords": None,
        "rope_jitter_coords": None,
        "rope_rescale_coords": None,
        "rope_base": _get(ck_args, "rope_base", 100.0),
        "foundation_stats_path": "weights/foundation_stats.pt",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ema_source", default="cascade_a_3")
    args = ap.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    base_state = ckpt.get("state_dict") or ckpt.get("model")
    if not isinstance(base_state, dict):
        raise RuntimeError("source checkpoint has no state_dict/model dict")
    recurrence = [
        key for key in base_state
        if "recurrent" in key.lower() or "state_attn" in key.lower()
    ]
    if recurrence:
        raise RuntimeError(
            "source checkpoint is not pre-recurrence; found recurrence keys: "
            + ", ".join(recurrence[:8])
        )

    selected, ema_meta = _parse_ema(ckpt, args.ema_source)
    state_dict = _normalize_state_dict(selected)
    if any("recurrent" in key.lower() or "state_attn" in key.lower()
           for key in state_dict):
        raise RuntimeError("selected state contains recurrence tensors")

    ck_args = _as_dict(ckpt.get("args"))
    try:
        source_ref = str(source.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        source_ref = source.name
    export = {
        "format": "corridorkey_v2_runtime_checkpoint",
        "format_version": 1,
        "state_dict": state_dict,
        "model_config": _model_config(ck_args),
        "metadata": {
            "public_model_name": "CorridorKey v2",
            "source_checkpoint": source_ref,
            "source_epoch": ckpt.get("epoch"),
            "source_global_step": ckpt.get("global_step"),
            "source_samples_seen": ckpt.get("samples_seen"),
            "ema_source": args.ema_source,
            "ema": ema_meta,
            "pre_recurrence": True,
            "shape_quantum": 128,
            "required_vfms": ["C-RADIOv4-SO400M", "MoGe-2-ViT-B", "RVM-small"],
            "torch_version_exported": torch.__version__,
        },
    }
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(export, tmp)
    os.replace(tmp, output)
    print(f"wrote {output} ({output.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
