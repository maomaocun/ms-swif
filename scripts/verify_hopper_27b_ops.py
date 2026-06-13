#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    seconds: float


def timed(name: str, fn: Callable[[], str]) -> CheckResult:
    start = time.perf_counter()
    try:
        return CheckResult(name, True, fn(), time.perf_counter() - start)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if os.environ.get("VERBOSE_TRACEBACK") == "1":
            detail += "\n" + traceback.format_exc()
        return CheckResult(name, False, detail, time.perf_counter() - start)


def format_bool(value: bool) -> str:
    return "OK" if value else "FAIL"


def get_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config


def load_model_config(model_path: str) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found under {model_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def check_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    lines = [f"torch={torch.__version__} cuda={torch.version.cuda} devices={torch.cuda.device_count()}"]
    for idx in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(idx)
        name = torch.cuda.get_device_name(idx)
        lines.append(f"gpu{idx}={name} sm={cap[0]}{cap[1]}")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (9, 0):
        raise RuntimeError(f"device 0 is not Hopper-class SM90: sm={major}{minor}")
    return "; ".join(lines)


def check_model_config(args: argparse.Namespace) -> str:
    config_path = Path(args.model_path) / "config.json"
    if not config_path.is_file():
        return f"skipped: config.json not found under {args.model_path}"
    cfg = load_model_config(args.model_path)
    text = get_text_config(cfg)
    hidden = int(text.get("hidden_size", 0))
    layers = int(text.get("num_hidden_layers", 0))
    vocab = int(text.get("vocab_size", 0))
    max_pos = int(text.get("max_position_embeddings", 0))
    layer_types = text.get("layer_types") or []
    full_layers = sum(1 for item in layer_types if item == "full_attention")
    linear_layers = sum(1 for item in layer_types if item == "linear_attention")

    if args.tp <= 0:
        raise ValueError("--tp must be positive")
    vocab_per_tp = math.ceil(vocab / args.tp) if vocab else 0
    checks = []
    if vocab and vocab % args.tp != 0:
        checks.append(f"vocab not divisible by TP: {vocab}/{args.tp}")
    linear_key_heads = text.get("linear_num_key_heads")
    if linear_key_heads is not None:
        divisor = args.tp * args.cp
        if int(linear_key_heads) % divisor != 0:
            checks.append(f"linear_num_key_heads={linear_key_heads} not divisible by TP*CP={divisor}")

    if checks:
        raise RuntimeError("; ".join(checks))
    return (
        f"hidden={hidden} layers={layers} vocab={vocab} vocab_per_tp={vocab_per_tp} "
        f"max_pos={max_pos} full_attention={full_layers} linear_attention={linear_layers} "
        f"tp={args.tp} cp={args.cp}"
    )


def check_flash_attn3(args: argparse.Namespace) -> str:
    torch.cuda.set_device(args.device)
    module = importlib.import_module("flash_attn_interface")
    config_detail = ""
    try:
        cfg = importlib.import_module("flash_attn_config")
        disabled = {
            name: getattr(cfg, name)
            for name in dir(cfg)
            if name.startswith("DISABLE_")
        }
        config_detail = " disabled=" + ",".join(f"{k}={v}" for k, v in sorted(disabled.items()))
    except Exception as exc:
        config_detail = f" config_unavailable={type(exc).__name__}:{exc}"

    q = torch.randn(args.batch, args.seq, args.heads, args.head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q) * 0.02
    v = torch.randn_like(q)
    torch.cuda.synchronize()
    out = module.flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    if out.shape != q.shape:
        raise RuntimeError(f"unexpected FA3 output shape {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        raise RuntimeError("FA3 output contains non-finite values")
    return f"module={module.__file__} out={tuple(out.shape)} dtype={out.dtype}{config_detail}"


def check_fla_gated_delta(args: argparse.Namespace) -> str:
    torch.cuda.set_device(args.device)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q = torch.randn(args.batch, args.seq, args.heads, args.head_dim, device="cuda", dtype=torch.bfloat16) * 0.001
    k = torch.randn_like(q) * 0.001
    v = torch.randn_like(q) * 0.001
    # GatedDeltaNet uses log-decay-like gates in real models. Keep this smoke
    # input conservative so the check validates kernel availability, not
    # behavior under arbitrary unstable gates.
    g = -torch.rand(args.batch, args.seq, args.heads, device="cuda", dtype=torch.float32) * 0.1
    beta = torch.full((args.batch, args.seq, args.heads), 0.5, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    out, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=args.head_dim**-0.5,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
    )
    torch.cuda.synchronize()
    if out.shape != q.shape:
        raise RuntimeError(f"unexpected FLA output shape {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        raise RuntimeError("FLA output contains non-finite values")
    final_shape = None if final_state is None else tuple(final_state.shape)
    return f"chunk_gated_delta_rule={chunk_gated_delta_rule.__module__} out={tuple(out.shape)} final={final_shape}"


def check_te_fp8_support() -> str:
    import transformer_engine as transformer_engine
    from transformer_engine.pytorch.fp8 import check_fp8_support

    supported, reason = check_fp8_support()
    if not supported:
        raise RuntimeError(reason or "Transformer Engine reported FP8 unsupported")
    return f"transformer_engine={getattr(transformer_engine, '__version__', '<unknown>')}"


def make_fp8_recipe(recipe_name: str):
    from transformer_engine.common import recipe

    if recipe_name == "delayed":
        return recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID, amax_history_len=1024, amax_compute_algo="max")
    if recipe_name == "mxfp8":
        return recipe.MXFP8BlockScaling(fp8_format=recipe.Format.E4M3)
    if recipe_name == "current":
        return recipe.Float8CurrentScaling(fp8_format=recipe.Format.HYBRID)
    raise ValueError(f"unsupported FP8 recipe: {recipe_name}")


def check_te_fp8_linear(args: argparse.Namespace) -> str:
    torch.cuda.set_device(args.device)
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import fp8_autocast

    x = torch.randn(args.seq, args.batch, args.hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    linear = te.Linear(args.hidden, args.hidden, params_dtype=torch.bfloat16, bias=False).cuda()
    fp8_recipe = make_fp8_recipe(args.fp8_recipe)
    torch.cuda.synchronize()
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        y = linear(x)
        loss = y.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(y).all():
        raise RuntimeError("TE FP8 Linear output contains non-finite values")
    return f"recipe={args.fp8_recipe} y={tuple(y.shape)} dtype={y.dtype} grad={x.grad.dtype}"


def check_te_fp8_mlp(args: argparse.Namespace) -> str:
    torch.cuda.set_device(args.device)
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import fp8_autocast

    x = torch.randn(args.seq, args.batch, args.hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mlp = te.LayerNormMLP(
        args.hidden,
        args.ffn_hidden,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        activation="swiglu",
        bias=False,
    ).cuda()
    fp8_recipe = make_fp8_recipe(args.fp8_recipe)
    torch.cuda.synchronize()
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        y = mlp(x)
        if isinstance(y, tuple):
            y = y[0]
        loss = y.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(y).all():
        raise RuntimeError("TE FP8 MLP output contains non-finite values")
    return f"recipe={args.fp8_recipe} y={tuple(y.shape)} dtype={y.dtype} grad={x.grad.dtype}"


def print_recommendations(args: argparse.Namespace) -> None:
    print("\nRecommended 27B Hopper A/B commands:")
    base = "TRAIN_ITERS=2 SAVE_STEPS=1000000 REPORT_TO=none"
    print(
        f"  {base} FP8_FORMAT=hybrid FP8_RECIPE={args.fp8_recipe} "
        "RUN_NAME=ab-hopper-fp8 bash train_qwen36_27b_paper2arm_distill_megatron.sh"
    )
    print(
        f"  {base} FP8_FORMAT=hybrid FP8_RECIPE={args.fp8_recipe} FP8_PARAM_GATHER=true "
        "RUN_NAME=ab-hopper-fp8-param-gather bash train_qwen36_27b_paper2arm_distill_megatron.sh"
    )
    print(
        "  "
        + base
        + " OVERLAP_CPU_OPTIMIZER_D2H_H2D=true TP_COMM_OVERLAP=true "
        "GRADIENT_ACCUMULATION_FUSION=true RUN_NAME=ab-hopper-overlap bash train_qwen36_27b_paper2arm_distill_megatron.sh"
    )
    print(
        "  "
        + base
        + " LINEAR_CE_CHUNK_SIZE=4096 RUN_NAME=ab-hopper-linear-ce4096 "
        "bash train_qwen36_27b_paper2arm_distill_megatron.sh"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Hopper/SM90 operators for Qwen3.6 27B Megatron SFT.")
    parser.add_argument("--model-path", default="/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--ffn-hidden", type=int, default=27648)
    parser.add_argument("--fp8-recipe", choices=["delayed", "mxfp8", "current"], default="delayed")
    parser.add_argument("--skip-fa3", action="store_true")
    parser.add_argument("--skip-fla", action="store_true")
    parser.add_argument("--skip-fp8", action="store_true")
    parser.add_argument("--recommendations", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[tuple[str, Callable[[], str]]] = [
        ("cuda_sm90", check_cuda),
        ("qwen36_27b_config", lambda: check_model_config(args)),
    ]
    if not args.skip_fa3:
        checks.append(("flash_attention_3", lambda: check_flash_attn3(args)))
    if not args.skip_fla:
        checks.append(("fla_gated_delta_rule", lambda: check_fla_gated_delta(args)))
    if not args.skip_fp8:
        checks.extend(
            [
                ("transformer_engine_fp8_support", check_te_fp8_support),
                ("transformer_engine_fp8_linear", lambda: check_te_fp8_linear(args)),
                ("transformer_engine_fp8_mlp_swiglu", lambda: check_te_fp8_mlp(args)),
            ]
        )

    results = [timed(name, fn) for name, fn in checks]
    width = max(len(result.name) for result in results)
    for result in results:
        print(f"[{format_bool(result.ok):4}] {result.name:<{width}} {result.seconds:7.2f}s  {result.detail}")

    if args.recommendations:
        print_recommendations(args)

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
