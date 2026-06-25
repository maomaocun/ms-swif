#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def reward_of(result: dict) -> float:
    verifier_result = result.get('verifier_result') or {}
    rewards = verifier_result.get('rewards') or {}
    reward = rewards.get('reward', 0.0)
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def observation_text(step: dict) -> str:
    observation = step.get('observation') or {}
    results = observation.get('results') or []
    chunks = []
    for result in results:
        content = result.get('content')
        if content:
            chunks.append(str(content))
    return '\n'.join(chunks)


def agent_step_content(step: dict) -> str:
    parts = []
    message = step.get('message')
    if message:
        parts.append(str(message).strip())

    tool_calls = step.get('tool_calls') or []
    for call in tool_calls:
        name = call.get('function_name') or call.get('name') or 'tool'
        arguments = call.get('arguments')
        parts.append(f'\n[tool_call:{name}]\n{json.dumps(arguments, ensure_ascii=False)}')

    observation = observation_text(step)
    if observation:
        parts.append(f'\n[tool_observation]\n{observation}')

    return '\n'.join(part for part in parts if part).strip()


def trajectory_to_messages(path: Path) -> list[dict[str, str]]:
    trajectory = read_json(path)
    messages = []
    for step in trajectory.get('steps') or []:
        source = step.get('source')
        if source == 'system':
            role = 'system'
            content = str(step.get('message') or '').strip()
        elif source == 'user':
            role = 'user'
            content = str(step.get('message') or '').strip()
        elif source in {'agent', 'assistant'}:
            role = 'assistant'
            content = agent_step_content(step)
        else:
            continue
        if content:
            messages.append({'role': role, 'content': content})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reward-threshold', default=1.0, type=float)
    parser.add_argument('--max-samples', default=0, type=int)
    args = parser.parse_args()

    written = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as out:
        for result_path in sorted(args.input_root.glob('batch_*/*/result.json')):
            result = read_json(result_path)
            if reward_of(result) < args.reward_threshold:
                skipped += 1
                continue
            trajectory_path = result_path.parent / 'agent' / 'trajectory.json'
            if not trajectory_path.exists():
                skipped += 1
                continue
            messages = trajectory_to_messages(trajectory_path)
            if len(messages) < 3:
                skipped += 1
                continue
            row = {
                'messages': messages,
                'source': 'ziyue_swe-smith-2k',
                'task_name': result.get('task_name'),
                'trial_name': result.get('trial_name'),
                'reward': reward_of(result),
            }
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            written += 1
            if args.max_samples and written >= args.max_samples:
                break

    print(f'wrote={written} skipped={skipped} output={args.output}')


if __name__ == '__main__':
    main()
