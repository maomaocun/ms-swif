#!/usr/bin/env python3
"""Measure full-trajectory paper2arm SFT token lengths with the real ms-swift template."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def percentile(sorted_values: list[int | float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * p / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def summarize(values: list[int | float]) -> dict[str, float]:
    vals = sorted(values)
    return {
        "count": float(len(vals)),
        "min": float(vals[0]) if vals else 0.0,
        "p05": percentile(vals, 5),
        "p10": percentile(vals, 10),
        "p25": percentile(vals, 25),
        "p50": percentile(vals, 50),
        "p75": percentile(vals, 75),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "max": float(vals[-1]) if vals else 0.0,
        "mean": float(statistics.mean(vals)) if vals else 0.0,
    }


def threshold_counts(values: list[int], thresholds: list[int]) -> dict[str, int]:
    return {f"le_{t}": sum(v <= t for v in values) for t in thresholds} | {
        f"gt_{t}": sum(v > t for v in values) for t in thresholds
    }


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def count_tool_calls(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "tool_call")


def count_chars(sample: dict[str, Any]) -> int:
    total = len(sample.get("tools") or "")
    for msg in sample.get("messages") or []:
        total += len(msg.get("content") or "")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", default="/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B")
    parser.add_argument("--agent-template", default="qwen3_5")
    parser.add_argument("--loss-scale", default="default+ignore_empty_think")
    parser.add_argument("--max-length", type=int, default=1_000_000)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    from swift import get_processor, get_template

    samples = load_samples(args.dataset)
    print(f"loading processor: {args.model}")
    processor = get_processor(args.model)
    template = get_template(
        processor,
        agent_template=args.agent_template,
        loss_scale=args.loss_scale,
        max_length=args.max_length,
        truncation_strategy="raise",
        preserve_thinking=True,
    )
    template.set_mode("train")

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        encoded = template.encode(sample)
        input_ids = encoded["input_ids"]
        labels = encoded.get("labels") or []
        trainable = sum(1 for x in labels if x != -100)
        input_tokens = len(input_ids)
        row = {
            "idx": idx,
            "uuid": sample.get("uuid"),
            "channel": sample.get("channel"),
            "reward": sample.get("metadata", {}).get("reward"),
            "score_0_100": sample.get("metadata", {}).get("score_0_100"),
            "messages": len(sample.get("messages") or []),
            "tool_calls": count_tool_calls(sample.get("messages") or []),
            "chars": count_chars(sample),
            "input_tokens": input_tokens,
            "trainable_tokens": trainable,
            "masked_tokens": input_tokens - trainable,
            "trainable_ratio": trainable / input_tokens if input_tokens else 0.0,
        }
        rows.append(row)
        print(
            f"[{idx:03d}/{len(samples):03d}] {row['uuid']} "
            f"tokens={input_tokens} trainable={trainable} reward={row['reward']}"
        )

    thresholds = [32768, 65536, 98304, 131072, 196608, 262144, 327680, 524288]
    input_tokens = [int(r["input_tokens"]) for r in rows]
    trainable_tokens = [int(r["trainable_tokens"]) for r in rows]
    chars = [int(r["chars"]) for r in rows]
    summary = {
        "dataset": str(args.dataset),
        "model": args.model,
        "agent_template": args.agent_template,
        "loss_scale": args.loss_scale,
        "max_length_for_measurement": args.max_length,
        "n_samples": len(rows),
        "input_tokens": summarize(input_tokens),
        "trainable_tokens": summarize(trainable_tokens),
        "chars": summarize(chars),
        "fit_counts": threshold_counts(input_tokens, thresholds),
        "longest": sorted(rows, key=lambda r: r["input_tokens"], reverse=True)[:10],
        "shortest": sorted(rows, key=lambda r: r["input_tokens"])[:10],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {args.output_json}")
    print(f"per-sample: {args.output_csv}")


if __name__ == "__main__":
    main()
