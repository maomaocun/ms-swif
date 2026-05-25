#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import time

import torch
import torch.nn.functional as F


def gib(num_bytes: int | float) -> float:
    return float(num_bytes) / 1024**3


def parse_dtype(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


class ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor, chunk_size: int):
        losses = torch.empty((hidden.shape[0],), device=hidden.device, dtype=torch.float32)

        for start in range(0, hidden.shape[0], chunk_size):
            end = min(hidden.shape[0], start + chunk_size)
            logits = hidden[start:end].matmul(weight.t()).float()
            losses[start:end] = F.cross_entropy(
                logits,
                target[start:end],
                ignore_index=-100,
                reduction="none",
            )

        ctx.save_for_backward(hidden, weight, target)
        ctx.chunk_size = chunk_size
        return losses

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, target = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        grad_hidden = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None

        for start in range(0, hidden.shape[0], chunk_size):
            end = min(hidden.shape[0], start + chunk_size)
            hidden_chunk = hidden[start:end]
            target_chunk = target[start:end]
            logits = hidden_chunk.matmul(weight.t()).float()
            grad_logits = torch.softmax(logits, dim=-1)

            valid = target_chunk.ne(-100)
            if valid.any():
                row_idx = torch.arange(end - start, device=hidden.device)
                safe_target = target_chunk.clamp_min(0)
                grad_logits[row_idx[valid], safe_target[valid]] -= 1.0
            grad_logits.masked_fill_(~valid.unsqueeze(-1), 0.0)
            grad_logits.mul_(grad_output[start:end].float().unsqueeze(-1))

            if grad_hidden is not None:
                grad_hidden[start:end] = grad_logits.matmul(weight.float()).to(hidden.dtype)
            if grad_weight is not None:
                grad_weight.add_(grad_logits.t().matmul(hidden_chunk.float()).to(weight.dtype))

        return grad_hidden, grad_weight, None, None


def make_inputs(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(args.seed)
    hidden = torch.empty((args.tokens, args.hidden), device=device, dtype=dtype, requires_grad=True)
    weight = torch.empty((args.vocab_shard, args.hidden), device=device, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        hidden.normal_(mean=0.0, std=0.02)
        weight.normal_(mean=0.0, std=1.0 / math.sqrt(args.hidden))

    target = torch.randint(0, args.vocab_shard, (args.tokens,), device=device, dtype=torch.long)
    if args.ignore_fraction > 0:
        mask = torch.rand((args.tokens,), device=device) < args.ignore_fraction
        target = target.masked_fill(mask, -100)
    return hidden, weight, target


def run_case(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = parse_dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    hidden, weight, target = make_inputs(args, dtype, device)
    valid_count = target.ne(-100).sum().clamp_min(1)
    torch.cuda.synchronize(device)

    base_alloc = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    if args.mode == "full":
        logits = hidden.matmul(weight.t())
        ce_logits = logits.float() if args.full_ce_fp32 else logits
        losses = F.cross_entropy(ce_logits, target, ignore_index=-100, reduction="none")
    elif args.mode == "chunked":
        losses = ChunkedLinearCE.apply(hidden, weight, target, args.chunk_size)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    loss = losses.sum() / valid_count
    if args.backward:
        loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    end_alloc = torch.cuda.memory_allocated(device)
    end_reserved = torch.cuda.memory_reserved(device)

    return {
        "mode": args.mode,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "vocab_shard": args.vocab_shard,
        "chunk_size": args.chunk_size if args.mode == "chunked" else None,
        "dtype": args.dtype,
        "full_ce_fp32": args.full_ce_fp32 if args.mode == "full" else None,
        "backward": args.backward,
        "ignore_fraction": args.ignore_fraction,
        "loss": float(loss.detach().cpu()),
        "elapsed_sec": elapsed,
        "base_alloc_gib": gib(base_alloc),
        "base_reserved_gib": gib(base_reserved),
        "peak_alloc_gib": gib(peak_alloc),
        "peak_reserved_gib": gib(peak_reserved),
        "incremental_peak_alloc_gib": gib(peak_alloc - base_alloc),
        "incremental_peak_reserved_gib": gib(peak_reserved - base_reserved),
        "end_alloc_gib": gib(end_alloc),
        "end_reserved_gib": gib(end_reserved),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "chunked"], required=True)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--vocab-shard", type=int, default=31040)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-fraction", type=float, default=0.0)
    parser.add_argument("--full-ce-fp32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        result = run_case(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        result = {
            "mode": args.mode,
            "tokens": args.tokens,
            "hidden": args.hidden,
            "vocab_shard": args.vocab_shard,
            "chunk_size": args.chunk_size if args.mode == "chunked" else None,
            "dtype": args.dtype,
            "backward": args.backward,
            "oom": True,
            "error": str(exc).splitlines()[0],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
