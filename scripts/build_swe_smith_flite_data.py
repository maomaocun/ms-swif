#!/usr/bin/env python3
"""Build SWE-Smith flite_data JSONL files from trajectory_filter splits.

The script joins three sources:

1. Raw Harbor trial directories under data/raw/swe-smith-2k.
2. trajectory_filter split records with quality_tier labels.
3. mini-swe-agent trajectory conversion logic used by ms-swift SFT data.

It writes several ms-swift-compatible JSONL files so training can compare
strict positive data against broader keep data without changing raw artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_paper2arm_qwen37_to_swift import (  # noqa: E402
    contains_provider_failure,
    convert_trial,
    load_trial_result_info,
    validate_sample,
    write_jsonl,
)


DATASET_SPECS: dict[str, set[str]] = {
    "positive_strict_only": {"positive_strict"},
    "positive_strict_clean": {"positive_strict", "positive_clean"},
    "positive_broad": {"positive_strict", "positive_clean", "positive_with_exit_warning"},
    "review_positive": {"positive_low_score_review", "positive_timeout_review"},
    "failed_prefix_negative": {"failed_prefix_or_negative"},
}


def load_split_records(filter_dir: Path) -> dict[str, dict[str, Any]]:
    by_trial: dict[str, dict[str, Any]] = {}
    for split_name in ("keep", "review", "drop"):
        split_path = filter_dir / "splits" / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split file: {split_path}")
        with split_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_filter_split"] = split_name
                trial_path = record.get("trial_path")
                trial_name = Path(trial_path).name if trial_path else None
                if not trial_name:
                    trial_name = record.get("trajectory_id")
                if trial_name:
                    by_trial[str(trial_name)] = record
    return by_trial


def iter_trial_dirs(raw_root: Path) -> Iterable[Path]:
    if (raw_root / "result.json").exists() and (raw_root / "agent").exists():
        yield raw_root
        return

    for batch_dir in sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("batch")):
        for trial_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
            if (trial_dir / "result.json").exists() and (trial_dir / "agent").exists():
                yield trial_dir


def batch_name(trial_dir: Path) -> str:
    parent = trial_dir.parent
    return parent.name if parent.name.startswith("batch") else "unknown"


def augment_sample(
    sample: dict[str, Any],
    *,
    trial_dir: Path,
    filter_record: dict[str, Any],
    filter_dir: Path,
    output_set: str,
) -> dict[str, Any]:
    metadata = dict(sample.get("metadata") or {})
    filtering = filter_record.get("filtering") or {}
    overall = filter_record.get("overall") or {}
    metadata.update(
        {
            "source_dataset": "swe-smith-2k",
            "source_batch": batch_name(trial_dir),
            "trial": trial_dir.name,
            "trial_path": str(trial_dir),
            "trajectory_path": str(trial_dir / "agent" / "mini-swe-agent.trajectory.json"),
            "filter_source_dir": str(filter_dir),
            "filter_output_set": output_set,
            "filter_split": filter_record.get("_filter_split"),
            "quality_tier": filter_record.get("quality_tier"),
            "filter_reward": filter_record.get("reward"),
            "filter_overall_label": overall.get("label"),
            "filter_overall_score": overall.get("score"),
            "filter_overall_severity": overall.get("severity"),
            "filter_recommended_action": filtering.get("recommended_action"),
            "filter_training_use": filtering.get("training_use"),
            "filter_top_issues": filter_record.get("top_issues") or [],
        }
    )
    sample = dict(sample)
    sample["metadata"] = metadata
    sample["uuid"] = trial_dir.name
    return sample


def write_stats(
    output_dir: Path,
    *,
    raw_root: Path,
    filter_dir: Path,
    stats: dict[str, Any],
    outputs: dict[str, list[dict[str, Any]]],
) -> None:
    dataset_stats: dict[str, Any] = {}
    for name, samples in outputs.items():
        tiers = Counter((sample.get("metadata") or {}).get("quality_tier") for sample in samples)
        batches = Counter((sample.get("metadata") or {}).get("source_batch") for sample in samples)
        roles: Counter[str] = Counter()
        loss_false = 0
        msg_counts: list[int] = []
        char_counts: list[int] = []
        for sample in samples:
            messages = sample.get("messages") or []
            msg_counts.append(len(messages))
            char_counts.append(sum(len(message.get("content") or "") for message in messages))
            for message in messages:
                role = message.get("role")
                roles[str(role)] += 1
                if message.get("loss") is False:
                    loss_false += 1
        dataset_stats[name] = {
            "samples": len(samples),
            "quality_tiers": dict(tiers),
            "batches": dict(batches),
            "roles": dict(roles),
            "loss_false_messages": loss_false,
            "messages_per_sample": summarize(msg_counts),
            "chars_per_sample": summarize(char_counts),
            "file": str(output_dir / f"swe_smith_{name}_sft.jsonl"),
        }

    payload = {
        "raw_root": str(raw_root),
        "filter_dir": str(filter_dir),
        "output_dir": str(output_dir),
        "dataset_specs": {name: sorted(tiers) for name, tiers in DATASET_SPECS.items()},
        "datasets": dataset_stats,
        "conversion": stats,
    }
    (output_dir / "swe_smith_flite_data.stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def summarize(values: list[int]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "min": float(values[0]),
        "p50": float(values[len(values) // 2]),
        "p90": float(values[int((len(values) - 1) * 0.9)]),
        "max": float(values[-1]),
        "mean": float(sum(values) / len(values)),
        "sum": float(sum(values)),
    }


def write_readme(output_dir: Path) -> None:
    lines = [
        "# SWE-Smith flite_data",
        "",
        "本目录由 `training/sft/ms-swift/scripts/build_swe_smith_flite_data.py` 生成。",
        "它把 `trajectory_filter/data_filter/swe-smith-2k/splits` 中的 keep/review/drop 与 raw trial 转换结果连接，输出 ms-swift 可读取的 SFT JSONL。",
        "",
        "## 文件",
        "",
        "- `swe_smith_positive_strict_only_sft.jsonl`：只包含 `positive_strict`。",
        "- `swe_smith_positive_strict_clean_sft.jsonl`：包含 `positive_strict` 和 `positive_clean`，建议作为干净基线。",
        "- `swe_smith_positive_broad_sft.jsonl`：在干净基线上加入 `positive_with_exit_warning`，建议作为覆盖率 ablation。",
        "- `swe_smith_review_positive_sft.jsonl`：低分或 timeout 正样本，只用于复核/降权实验。",
        "- `swe_smith_failed_prefix_negative_sft.jsonl`：失败前缀或负样本池，不建议直接作为正向 SFT。",
        "- `swe_smith_flite_data.stats.json`：生成统计、跳过原因、每个输出集的 tier/batch/role 统计。",
        "",
        "## 默认训练建议",
        "",
        "先训练 `positive_strict_clean`，再训练 `positive_broad` 做 ablation；`review_positive` 和 `failed_prefix_negative` 需要额外 judge 或人工确认后再进入训练。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--filter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--teacher", default="deepseek-v4-pro")
    parser.add_argument("--drop-provider-failures", action="store_true")
    parser.add_argument("--mask-failed-tool-calls", action="store_true", default=True)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    filter_dir = args.filter_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_trial = load_split_records(filter_dir)
    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASET_SPECS}
    stats: dict[str, Any] = {
        "trial_count": 0,
        "matched_filter_record": 0,
        "converted": 0,
        "skipped_no_filter_record": 0,
        "skipped_unselected_tier": 0,
        "skipped_provider_failure": 0,
        "skipped_no_reward_info": 0,
        "skipped_invalid_or_missing_trajectory": 0,
        "skipped_validation_issue": 0,
        "validation_issues": [],
        "tier_counts_seen": {},
        "tier_counts_converted": {},
    }
    tier_seen: Counter[str] = Counter()
    tier_converted: Counter[str] = Counter()

    for trial_dir in iter_trial_dirs(raw_root):
        stats["trial_count"] += 1
        record = records_by_trial.get(trial_dir.name)
        if record is None:
            stats["skipped_no_filter_record"] += 1
            continue
        stats["matched_filter_record"] += 1
        tier = str(record.get("quality_tier") or "")
        tier_seen[tier] += 1
        target_sets = [name for name, tiers in DATASET_SPECS.items() if tier in tiers]
        if not target_sets:
            stats["skipped_unselected_tier"] += 1
            continue
        trajectory_path = trial_dir / "agent" / "mini-swe-agent.trajectory.json"
        if args.drop_provider_failures and trajectory_path.exists() and contains_provider_failure(trajectory_path):
            stats["skipped_provider_failure"] += 1
            continue
        reward_info = load_trial_result_info(trial_dir)
        if reward_info is None:
            stats["skipped_no_reward_info"] += 1
        sample = convert_trial(
            trial_dir,
            reward_info,
            teacher=args.teacher,
            mask_tool_calls=False,
            mask_failed_tool_calls=args.mask_failed_tool_calls,
        )
        if sample is None:
            stats["skipped_invalid_or_missing_trajectory"] += 1
            continue
        issues = validate_sample(sample)
        if issues:
            stats["skipped_validation_issue"] += 1
            if len(stats["validation_issues"]) < 30:
                stats["validation_issues"].append({"trial": trial_dir.name, "issues": issues})
            continue
        stats["converted"] += 1
        tier_converted[tier] += 1
        for output_set in target_sets:
            outputs[output_set].append(
                augment_sample(
                    sample,
                    trial_dir=trial_dir,
                    filter_record=record,
                    filter_dir=filter_dir,
                    output_set=output_set,
                )
            )

    stats["tier_counts_seen"] = dict(tier_seen)
    stats["tier_counts_converted"] = dict(tier_converted)

    for name, samples in outputs.items():
        write_jsonl(samples, output_dir / f"swe_smith_{name}_sft.jsonl")
    write_stats(output_dir, raw_root=raw_root, filter_dir=filter_dir, stats=stats, outputs=outputs)
    write_readme(output_dir)

    print(json.dumps({"output_dir": str(output_dir), "conversion": stats, "outputs": {k: len(v) for k, v in outputs.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
