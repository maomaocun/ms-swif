#!/usr/bin/env python3
"""Convert synthetic SFT JSONL with reasoning_content to ms-swift chat JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def format_assistant_content(content: str, reasoning: str) -> str:
    if "<think>" in content and "</think>" in content:
        return content
    if reasoning and content and reasoning == content:
        # Harbor issue #26 style duplicate: keep one copy only.
        reasoning = ""
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{content}" if content else f"<think>\n{reasoning}\n</think>\n"
    if content:
        return f"<think>\n\n</think>\n\n{content}"
    return "<think>\n\n</think>\n"


def quality_channel(reward: Any) -> str:
    if not isinstance(reward, (int, float)):
        return "unknown"
    value = float(reward)
    if 0 <= value <= 1:
        if value >= 0.8:
            return "reward_ge_0.8"
        if value >= 0.6:
            return "reward_0.6_0.8"
        if value >= 0.5:
            return "reward_0.5_0.6"
        return "reward_lt_0.5"
    if value >= 4:
        return "score_ge_4"
    if value >= 3:
        return "score_ge_3"
    if value >= 2:
        return "score_ge_2"
    return "score_lt_2"


def convert_sample(sample: dict[str, Any], idx: int) -> dict[str, Any] | None:
    src_messages = sample.get("messages")
    if not isinstance(src_messages, list):
        return None

    metadata = sample.get("synthetic_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    problem_id = normalize_text(metadata.get("problem_id"))
    uuid = problem_id or f"synthetic-{idx:06d}"

    messages: list[dict[str, Any]] = []
    for src in src_messages:
        if not isinstance(src, dict):
            continue
        role = src.get("role")
        content = normalize_text(src.get("content"))
        if role == "assistant":
            reasoning = normalize_text(src.get("reasoning_content"))
            messages.append({"role": "assistant", "content": format_assistant_content(content, reasoning)})
        elif role in {"system", "user"}:
            if content:
                messages.append({"role": role, "content": content})

    if len(messages) < 3 or not any(msg.get("role") == "assistant" for msg in messages):
        return None

    return {
        "messages": messages,
        "channel": quality_channel(metadata.get("reward")),
        "uuid": uuid,
        "metadata": {"source": "synthetic_sft", **metadata},
    }


def summarize(values: list[int | float]) -> dict[str, float] | None:
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    return {
        "min": vals[0],
        "p50": vals[len(vals) // 2],
        "p90": vals[int((len(vals) - 1) * 0.9)],
        "max": vals[-1],
        "mean": statistics.mean(vals),
        "sum": sum(vals),
    }


def write_stats(samples: list[dict[str, Any]], stats_path: Path, input_path: Path) -> None:
    role_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    msg_counts: list[int] = []
    char_counts: list[int] = []
    assistant_counts: list[int] = []
    rewards: list[float] = []
    for sample in samples:
        messages = sample.get("messages") or []
        msg_counts.append(len(messages))
        channel_counts[sample.get("channel", "unknown")] += 1
        reward = sample.get("metadata", {}).get("reward")
        if isinstance(reward, (int, float)):
            rewards.append(float(reward))
        total_chars = 0
        assistant_chars = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            role_counts[role] += 1
            total_chars += len(content)
            if role == "assistant":
                assistant_chars += len(content)
        char_counts.append(total_chars)
        assistant_counts.append(assistant_chars)

    stats = {
        "source_file": str(input_path),
        "n_samples": len(samples),
        "roles": dict(role_counts),
        "channels": dict(channel_counts),
        "messages_per_sample": summarize(msg_counts),
        "chars_per_sample": summarize(char_counts),
        "assistant_chars_per_sample": summarize(assistant_counts),
        "reward": summarize(rewards),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--stats-file", type=Path)
    args = parser.parse_args()

    samples: list[dict[str, Any]] = []
    skipped = 0
    with args.input_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            sample = convert_sample(json.loads(line), idx)
            if sample is None:
                skipped += 1
                continue
            samples.append(sample)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats_path = args.stats_file or args.output_file.with_suffix(args.output_file.suffix + ".stats.json")
    write_stats(samples, stats_path, args.input_file)

    print(f"input_file: {args.input_file}")
    print(f"output_file: {args.output_file}")
    print(f"stats_file: {stats_path}")
    print(f"samples: {len(samples)}")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
