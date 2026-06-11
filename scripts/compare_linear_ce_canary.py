#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


_ELAPSED_PART_RE = re.compile(r"(\d+)([hms])")


def _find_logging_jsonl(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "logging.jsonl"
    if candidate.is_file():
        return candidate
    matches = sorted(path.glob("**/logging.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No logging.jsonl found under {path}")
    raise ValueError(f"Multiple logging.jsonl files under {path}; pass one explicitly")


def _elapsed_to_seconds(value: str) -> float:
    total = 0
    for amount, unit in _ELAPSED_PART_RE.findall(value.replace(" ", "")):
        if unit == "h":
            total += int(amount) * 3600
        elif unit == "m":
            total += int(amount) * 60
        elif unit == "s":
            total += int(amount)
    return float(total)


def _parse_step(row: dict[str, Any]) -> int | None:
    iteration = row.get("iteration")
    if not isinstance(iteration, str) or "/" not in iteration:
        return None
    step, _ = iteration.split("/", 1)
    try:
        return int(step)
    except ValueError:
        return None


def _load_steps(path: Path) -> list[dict[str, Any]]:
    logging_path = _find_logging_jsonl(path)
    steps: list[dict[str, Any]] = []
    for line in logging_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "loss" not in row or "memory(GiB)" not in row:
            continue
        step = _parse_step(row)
        if step is None:
            continue
        row["_step"] = step
        row["_elapsed_sec"] = _elapsed_to_seconds(str(row.get("elapsed_time", "")))
        steps.append(row)
    if not steps:
        raise ValueError(f"No train step rows found in {logging_path}")
    return steps


def _incremental_step_times(steps: list[dict[str, Any]]) -> dict[int, float]:
    result: dict[int, float] = {}
    previous_elapsed = 0.0
    previous_step = 0
    for row in sorted(steps, key=lambda item: item["_step"]):
        step = int(row["_step"])
        elapsed = float(row["_elapsed_sec"])
        if step <= previous_step:
            continue
        result[step] = (elapsed - previous_elapsed) / (step - previous_step)
        previous_elapsed = elapsed
        previous_step = step
    return result


def _max_abs(values: list[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def compare(native_path: Path, streaming_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    native_steps = _load_steps(native_path)
    streaming_steps = _load_steps(streaming_path)
    native_by_step = {int(row["_step"]): row for row in native_steps}
    streaming_by_step = {int(row["_step"]): row for row in streaming_steps}
    common_steps = sorted(set(native_by_step) & set(streaming_by_step))
    if len(common_steps) < args.min_steps:
        raise ValueError(f"Need at least {args.min_steps} common steps, got {common_steps}")

    loss_deltas = [
        float(streaming_by_step[step]["loss"]) - float(native_by_step[step]["loss"])
        for step in common_steps
    ]
    grad_deltas = [
        float(streaming_by_step[step].get("grad_norm", 0.0)) - float(native_by_step[step].get("grad_norm", 0.0))
        for step in common_steps
    ]
    native_memory = max(float(row["memory(GiB)"]) for row in native_steps)
    streaming_memory = max(float(row["memory(GiB)"]) for row in streaming_steps)
    memory_saving = native_memory - streaming_memory

    native_times = _incremental_step_times(native_steps)
    streaming_times = _incremental_step_times(streaming_steps)
    speed_steps = [step for step in common_steps if not (args.skip_first_step and step == common_steps[0])]
    native_avg_time = _mean([native_times[step] for step in speed_steps if step in native_times])
    streaming_avg_time = _mean([streaming_times[step] for step in speed_steps if step in streaming_times])
    speed_regression = (
        (streaming_avg_time - native_avg_time) / native_avg_time
        if native_avg_time and not math.isnan(native_avg_time)
        else math.nan
    )

    checks = {
        "loss_abs": _max_abs(loss_deltas) <= args.loss_atol,
        "grad_norm_abs": _max_abs(grad_deltas) <= args.grad_norm_atol,
        "memory": memory_saving >= args.min_memory_saving_gib,
        "speed": math.isnan(speed_regression) or speed_regression <= args.max_speed_regression,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "common_steps": common_steps,
        "native_memory_gib": native_memory,
        "streaming_memory_gib": streaming_memory,
        "memory_saving_gib": memory_saving,
        "max_loss_abs_delta": _max_abs(loss_deltas),
        "max_grad_norm_abs_delta": _max_abs(grad_deltas),
        "native_avg_step_sec": native_avg_time,
        "streaming_avg_step_sec": streaming_avg_time,
        "speed_regression": speed_regression,
        "thresholds": {
            "loss_atol": args.loss_atol,
            "grad_norm_atol": args.grad_norm_atol,
            "min_memory_saving_gib": args.min_memory_saving_gib,
            "max_speed_regression": args.max_speed_regression,
            "min_steps": args.min_steps,
            "skip_first_step": args.skip_first_step,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare native vs streaming linear CE canary logs.")
    parser.add_argument("native_log", type=Path, help="Native run log dir or logging.jsonl")
    parser.add_argument("streaming_log", type=Path, help="Streaming run log dir or logging.jsonl")
    parser.add_argument("--loss-atol", type=float, default=2e-6)
    parser.add_argument("--grad-norm-atol", type=float, default=0.01)
    parser.add_argument("--min-memory-saving-gib", type=float, default=0.3)
    parser.add_argument("--max-speed-regression", type=float, default=0.35)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--skip-first-step", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    result = compare(args.native_log, args.streaming_log, args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
