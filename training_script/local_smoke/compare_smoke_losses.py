#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


METRIC_RE = re.compile(r"\{[^{}]*'loss'[^{}]*\}")


def latest_log(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no logs match {root / pattern}")
    path = matches[-1]
    return path / "train.log" if path.is_dir() else path


def resolve_log(value: str | None, root: Path, pattern: str) -> Path:
    if value:
        path = Path(value)
        return path / "train.log" if path.is_dir() else path
    return latest_log(root, pattern)


def resolve_bf16_log(value: str | None, root: Path) -> Path:
    if value:
        return resolve_log(value, root, "")
    patterns = [
        "qwen36-27b-paper2arm-distill-megatron-tp8-smoke-bf16-*/train.log",
        "qwen36-27b-paper2arm-distill-megatron-tp8-smoke-20*/train.log",
    ]
    last_error: Exception | None = None
    for pattern in patterns:
        try:
            return latest_log(root, pattern)
        except FileNotFoundError as exc:
            last_error = exc
    raise last_error or FileNotFoundError(f"no BF16 logs found under {root}")


def read_last_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = METRIC_RE.findall(text)
    if not matches:
        raise ValueError(f"no printed metrics dict with loss found in {path}")
    return ast.literal_eval(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Qwen3.6 27B smoke BF16 and FP8 losses.")
    parser.add_argument("--log-root", default="/model_cache/smoke_logs")
    parser.add_argument("--bf16-log", help="BF16 smoke log file or run directory")
    parser.add_argument("--fp8-log", help="FP8 smoke log file or run directory")
    args = parser.parse_args()

    root = Path(args.log_root)
    bf16_log = resolve_bf16_log(args.bf16_log, root)
    fp8_log = resolve_log(
        args.fp8_log,
        root,
        "qwen36-27b-paper2arm-distill-megatron-tp8-smoke-fp8-*/train.log",
    )

    bf16 = read_last_metrics(bf16_log)
    fp8 = read_last_metrics(fp8_log)
    bf16_loss = float(bf16["loss"])
    fp8_loss = float(fp8["loss"])
    delta = fp8_loss - bf16_loss
    rel = delta / bf16_loss if bf16_loss else float("nan")

    print(f"BF16 log: {bf16_log}")
    print(f"FP8 log:  {fp8_log}")
    print(f"BF16 loss: {bf16_loss:.8f}")
    print(f"FP8 loss:  {fp8_loss:.8f}")
    print(f"Delta:     {delta:+.8f} ({rel:+.2%})")
    if "train_speed(s/it)" in bf16 and "train_speed(s/it)" in fp8:
        print(f"BF16 speed: {float(bf16['train_speed(s/it)']):.6f} s/it")
        print(f"FP8 speed:  {float(fp8['train_speed(s/it)']):.6f} s/it")


if __name__ == "__main__":
    main()
