#!/usr/bin/env python3
"""Check Harbor #26 style duplicated reasoning/content in raw and Swift data.

The report intentionally avoids printing message text because raw trajectories
can contain credentials or private tool output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


THINK_RE = re.compile(r"^\s*<think>\n?(.*?)\n?</think>\s*(.*)\s*$", re.DOTALL)


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def event(path: Path, kind: str, detail: str, text: str, *, line: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "kind": kind,
        "detail": detail,
        "chars": len(text),
        "sha256_12": text_hash(text),
    }
    if line is not None:
        result["line"] = line
    return result


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(obj, dict):
                yield line_no, obj


def check_atif(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    for idx, step in enumerate(obj.get("steps") or []):
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        message = normalize_text(step.get("message"))
        reasoning = normalize_text(step.get("reasoning_content"))
        if message and reasoning and message == reasoning:
            findings.append(event(path, "raw_atif_equal", f"step={idx}", message))
    return findings


def check_mini(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    for idx, message_obj in enumerate(obj.get("messages") or []):
        if not isinstance(message_obj, dict) or message_obj.get("role") != "assistant":
            continue
        content = normalize_text(message_obj.get("content"))
        reasoning = normalize_text(message_obj.get("reasoning_content"))
        if content and reasoning and content == reasoning:
            findings.append(event(path, "raw_mini_equal", f"message={idx}", content))
    return findings


def check_swift_jsonl(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for line_no, sample in iter_jsonl(path):
        uuid = normalize_text(sample.get("uuid") or sample.get("metadata", {}).get("trial"))
        for idx, message_obj in enumerate(sample.get("messages") or []):
            if not isinstance(message_obj, dict) or message_obj.get("role") != "assistant":
                continue
            content = normalize_text(message_obj.get("content"))
            match = THINK_RE.match(content)
            if not match:
                continue
            thinking = normalize_text(match.group(1))
            outside = normalize_text(match.group(2))
            if thinking and outside and thinking == outside:
                detail = f"message={idx}"
                if uuid:
                    detail = f"uuid={uuid} {detail}"
                findings.append(event(path, "swift_think_equals_content", detail, thinking, line=line_no))
    return findings


def find_inputs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = [
                *path.rglob("trajectory.json"),
                *path.rglob("mini-swe-agent.trajectory.json"),
                *path.rglob("*.jsonl"),
            ]
        else:
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                found.append(candidate)
                seen.add(resolved)
    return sorted(found)


def check_path(path: Path) -> list[dict[str, Any]]:
    name = path.name
    if name == "trajectory.json":
        return check_atif(path)
    if name == "mini-swe-agent.trajectory.json":
        return check_mini(path)
    if path.suffix == ".jsonl":
        return check_swift_jsonl(path)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to scan.")
    parser.add_argument("--max-report", type=int, default=50, help="Maximum findings to print.")
    args = parser.parse_args()

    inputs = find_inputs(args.paths)
    findings: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in inputs:
        try:
            findings.extend(check_path(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")

    counts = Counter(finding["kind"] for finding in findings)
    print(f"files_scanned: {len(inputs)}")
    print(f"findings_total: {len(findings)}")
    for kind, count in sorted(counts.items()):
        print(f"{kind}: {count}")
    if errors:
        print(f"errors: {len(errors)}", file=sys.stderr)
        for err in errors[: args.max_report]:
            print(f"  {err}", file=sys.stderr)
    if findings:
        print("findings:")
        for finding in findings[: args.max_report]:
            print(json.dumps(finding, ensure_ascii=False, sort_keys=True))
        if len(findings) > args.max_report:
            print(f"... {len(findings) - args.max_report} more")
    return 1 if findings or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
