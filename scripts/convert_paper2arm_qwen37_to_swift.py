#!/usr/bin/env python3
"""Convert paper2arm Qwen3.7-max mini-swe-agent trajectories to ms-swift agent SFT JSONL.

The Qwen3.7-max run stores the real assistant actions in
agent/mini-swe-agent.trajectory.json. The ATIF trajectory.json keeps most
assistant messages empty and puts actions in tool_calls, so this converter uses
mini-swe-agent.trajectory.json as the authoritative source.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command inside the paper2arm sandbox and return stdout, stderr, return code, or execution exception information.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute. Use this to create files, run scripts, inspect outputs, and assemble the ARM submission.",
                }
            },
            "required": ["command"],
        },
    },
}


def load_reward_rows(run_dir: Path) -> list[dict[str, Any]]:
    candidate_names = [
        "verift_result.json",
        "verify_results_fixed_rubric_20260525.json",
        "verify_result_rerun.json",
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for name in candidate_names:
        path = run_dir / name
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    for pattern in ("verify_results*.json", "verify_result*.json"):
        for path in sorted(run_dir.glob(pattern)):
            if path.exists() and path not in seen:
                paths.append(path)
                seen.add(path)

    rows: list[dict[str, Any]] = []
    for path in paths:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            continue
        rows.extend(row for row in loaded if isinstance(row, dict))
    return rows


def load_reward_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = load_reward_rows(run_dir)
    reward_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("trial") and str(row["trial"]) not in reward_map:
            reward_map[str(row["trial"])] = row
    return reward_map


def parse_tool_call(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    function = tool_call.get("function") or {}
    name = function.get("name") or tool_call.get("function_name")
    raw_args = function.get("arguments")
    if raw_args is None and "arguments" in tool_call:
        raw_args = tool_call["arguments"]
    if not name:
        return None
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"command": raw_args} if name == "bash" else {"raw_arguments": raw_args}
    elif isinstance(raw_args, dict):
        args = raw_args
    elif raw_args is None:
        args = {}
    else:
        args = {"raw_arguments": raw_args}
    return {"name": name, "arguments": args}


def assistant_content(message: dict[str, Any]) -> str:
    reasoning = str(message.get("reasoning_content") or "").strip()
    content = str(message.get("content") or "").strip()
    if reasoning:
        if content:
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        return f"<think>\n{reasoning}\n</think>\n"
    if content:
        # Preserve non-thinking final answers while keeping Qwen thinking format explicit.
        return f"<think>\n\n</think>\n\n{content}"
    return "<think>\n\n</think>\n"


def normalize_tool_response(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def convert_messages(src_messages: list[dict[str, Any]], *, mask_tool_calls: bool = False) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for src in src_messages:
        role = src.get("role")
        if role == "system":
            messages.append({"role": "system", "content": str(src.get("content") or "")})
            continue
        if role == "user":
            content = str(src.get("content") or "")
            if not any(msg["role"] == "user" for msg in messages):
                messages.append({"role": "user", "content": content})
            else:
                # Mid-trajectory user messages are environment feedback from mini-swe-agent,
                # for example mandatory-tool-call errors. They are context, not a new task.
                messages.append({"role": "tool_response", "content": content})
            continue
        if role == "assistant":
            content = assistant_content(src)
            tool_calls = [tc for tc in (parse_tool_call(tc) for tc in src.get("tool_calls") or []) if tc]
            if content.strip() or tool_calls:
                messages.append({"role": "assistant", "content": content})
            for tool_call in tool_calls:
                message = {"role": "tool_call", "content": json.dumps(tool_call, ensure_ascii=False)}
                if mask_tool_calls:
                    message["loss"] = False
                messages.append(message)
            continue
        if role in {"tool", "tool_response"}:
            messages.append({"role": "tool_response", "content": normalize_tool_response(src.get("content"))})
            continue
        # mini-swe-agent writes a final exit message. It is runtime bookkeeping, not model behavior.
        if role == "exit":
            continue
    return compact_messages(messages)


def compact_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove empty bookkeeping and keep a valid ms-swift agent sequence."""
    cleaned: list[dict[str, str]] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role in {"system", "user"} and not content.strip():
            continue
        if role == "assistant" and content == "<think>\n\n</think>\n":
            # Keep the assistant turn only if followed by a tool_call; this is handled by not dropping here.
            pass
        cleaned.append({"role": role, "content": content})
    return cleaned


def quality_channel(reward: float | None) -> str:
    if reward is None:
        return "unknown"
    if reward >= 0.8:
        return "reward_ge_0.8"
    if reward >= 0.6:
        return "reward_0.6_0.8"
    if reward >= 0.5:
        return "reward_0.5_0.6"
    return "reward_lt_0.5"


def convert_trial(
    trial_dir: Path,
    reward_info: dict[str, Any] | None,
    *,
    teacher: str = "qwen3.7-max",
    mask_tool_calls: bool = False,
) -> dict[str, Any] | None:
    trajectory_path = trial_dir / "agent" / "mini-swe-agent.trajectory.json"
    if not trajectory_path.exists():
        return None
    obj = json.loads(trajectory_path.read_text(encoding="utf-8"))
    messages = convert_messages(obj.get("messages") or [], mask_tool_calls=mask_tool_calls)
    if len(messages) < 3:
        return None
    if not any(msg["role"] == "assistant" for msg in messages):
        return None
    if not any(msg["role"] == "tool_call" for msg in messages):
        return None
    reward = None
    score = None
    if reward_info:
        raw_reward = reward_info.get("reward")
        raw_score = reward_info.get("score_0_100")
        if isinstance(raw_reward, (int, float)):
            reward = float(raw_reward)
        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
    return {
        "tools": json.dumps([BASH_TOOL], ensure_ascii=False),
        "messages": messages,
        "channel": quality_channel(reward),
        "uuid": trial_dir.name,
        "metadata": {
            "trial": trial_dir.name,
            "reward": reward,
            "score_0_100": score,
            "teacher": teacher,
        },
    }


def validate_sample(sample: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    messages = sample.get("messages") or []
    if not messages:
        return ["empty messages"]
    if messages[0].get("role") != "system":
        issues.append("first message is not system")
    if len(messages) < 2 or messages[1].get("role") != "user":
        issues.append("second message is not user")
    valid_roles = {"system", "user", "assistant", "tool_call", "tool_response"}
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in valid_roles:
            issues.append(f"invalid role at {i}: {role}")
        if "content" not in msg:
            issues.append(f"missing content at {i}")
        if role == "tool_call":
            try:
                parsed = json.loads(msg.get("content") or "")
                if not isinstance(parsed, dict) or "name" not in parsed or "arguments" not in parsed:
                    issues.append(f"bad tool_call payload at {i}")
            except json.JSONDecodeError:
                issues.append(f"tool_call content is not JSON at {i}")
        if role == "tool_response" and i > 0 and messages[i - 1].get("role") not in {"tool_call", "tool_response"}:
            issues.append(f"tool_response at {i} not after tool_call/tool_response")
    return issues


def write_jsonl(samples: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_stats(samples: list[dict[str, Any]], stats_file: Path, run_dir: Path, min_reward: float | None) -> None:
    msg_counts = [len(s["messages"]) for s in samples]
    role_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    char_counts: list[int] = []
    assistant_chars: list[int] = []
    tool_call_counts: list[int] = []
    rewards: list[float] = []
    for sample in samples:
        channel_counts[sample.get("channel", "unknown")] += 1
        reward = sample.get("metadata", {}).get("reward")
        if isinstance(reward, (int, float)):
            rewards.append(float(reward))
        total_chars = 0
        asst_chars = 0
        tool_calls = 0
        for msg in sample["messages"]:
            role_counts[msg["role"]] += 1
            total_chars += len(msg.get("content", ""))
            if msg["role"] == "assistant":
                asst_chars += len(msg.get("content", ""))
            if msg["role"] == "tool_call":
                tool_calls += 1
        char_counts.append(total_chars)
        assistant_chars.append(asst_chars)
        tool_call_counts.append(tool_calls)
    stats = {
        "source_run_dir": str(run_dir),
        "min_reward": min_reward,
        "n_samples": len(samples),
        "roles": dict(role_counts),
        "channels": dict(channel_counts),
        "messages_per_sample": summarize(msg_counts),
        "chars_per_sample": summarize(char_counts),
        "assistant_chars_per_sample": summarize(assistant_chars),
        "tool_calls_per_sample": summarize(tool_call_counts),
        "reward": summarize(rewards),
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(values: list[float] | list[int]) -> dict[str, float] | None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--stats-file", type=Path)
    parser.add_argument("--min-reward", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--teacher", default="qwen3.7-max")
    parser.add_argument(
        "--mask-tool-calls",
        action="store_true",
        help="Set loss=false on tool_call messages. ms-swift otherwise renders tool_call as assistant output and trains it.",
    )
    args = parser.parse_args()

    reward_map = load_reward_map(args.input_dir)
    trial_dirs = sorted(p for p in args.input_dir.iterdir() if p.is_dir())
    samples: list[dict[str, Any]] = []
    skipped_low_reward = 0
    skipped_invalid = 0
    issues: list[tuple[str, list[str]]] = []

    for trial_dir in trial_dirs:
        reward_info = reward_map.get(trial_dir.name)
        reward = reward_info.get("reward") if reward_info else None
        if args.min_reward is not None and isinstance(reward, (int, float)) and float(reward) < args.min_reward:
            skipped_low_reward += 1
            continue
        sample = convert_trial(trial_dir, reward_info, teacher=args.teacher, mask_tool_calls=args.mask_tool_calls)
        if sample is None:
            skipped_invalid += 1
            continue
        sample_issues = validate_sample(sample)
        if sample_issues:
            issues.append((trial_dir.name, sample_issues))
            skipped_invalid += 1
            continue
        samples.append(sample)
        if args.max_samples and len(samples) >= args.max_samples:
            break

    write_jsonl(samples, args.output_file)
    stats_file = args.stats_file or args.output_file.with_suffix(args.output_file.suffix + ".stats.json")
    write_stats(samples, stats_file, args.input_dir, args.min_reward)

    print(f"input_dir: {args.input_dir}")
    print(f"trials: {len(trial_dirs)}")
    print(f"samples: {len(samples)}")
    print(f"skipped_low_reward: {skipped_low_reward}")
    print(f"skipped_invalid: {skipped_invalid}")
    print(f"output_file: {args.output_file}")
    print(f"stats_file: {stats_file}")
    if issues:
        print("issues:")
        for trial, trial_issues in issues[:20]:
            print(f"  {trial}: {trial_issues}")


if __name__ == "__main__":
    main()
