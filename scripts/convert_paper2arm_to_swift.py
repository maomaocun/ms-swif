#!/usr/bin/env python3
"""
Convert paper2arm trajectory.json files to ms-swift training format (OpenAI-style messages).

Loss mask strategy:
- system / user / tool messages: NO loss (masked by ms-swift default loss_scale)
- assistant messages: FULL loss (thinking + action + response)

Qwen3.5/3.6 thinking format:
- Assistant content is wrapped as: <think>\n{thinking}\n</think>\n\n{action_or_response}
- If no thinking exists, use: <think>\n\n</think>\n\n{action_or_response}
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def extract_thinking_and_action(message: str) -> tuple[str, str]:
    """
    Extract thinking and action/response from agent message.
    
    Returns (thinking, action_or_response).
    """
    message = message.strip()
    
    if message.startswith('THOUGHT:'):
        # Try to split at first code block (use lookahead to preserve ``` in rest)
        match = re.search(r'^THOUGHT:(.*?)(?=\n```|\Z)', message, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            rest_start = match.end()
            rest = message[rest_start:].strip()
            return thinking, rest
        else:
            # Pure thinking, no action
            thinking = message[len('THOUGHT:'):].strip()
            return thinking, ''
    
    # No THOUGHT prefix
    return '', message


def format_assistant_content(thinking: str, action_or_response: str) -> str:
    """Format assistant content for Qwen3.5/3.6 chat template."""
    if thinking:
        if action_or_response:
            return f'<think>\n{thinking}\n</think>\n\n{action_or_response}'
        else:
            return f'<think>\n{thinking}\n</think>\n\n'
    else:
        if action_or_response:
            return f'<think>\n\n</think>\n\n{action_or_response}'
        else:
            return '<think>\n\n</think>\n\n'


def observation_to_text(observation: Dict[str, Any]) -> str:
    """Convert observation dict to text content for tool message."""
    if not observation:
        return ''
    
    results = observation.get('results', [])
    parts = []
    for result in results:
        if isinstance(result, dict) and 'content' in result:
            parts.append(str(result['content']))
        else:
            parts.append(str(result))
    
    return '\n'.join(parts)


def trajectory_to_messages(trajectory: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Convert a single trajectory.json to ms-swift messages format."""
    steps = trajectory.get('steps', [])
    if not steps:
        return None
    
    messages = []
    
    for step in steps:
        source = step.get('source')
        
        if source == 'system':
            messages.append({
                'role': 'system',
                'content': step.get('message', '')
            })
        
        elif source == 'user':
            messages.append({
                'role': 'user',
                'content': step.get('message', '')
            })
        
        elif source == 'agent':
            msg = step.get('message', '')
            thinking, action_or_response = extract_thinking_and_action(msg)
            content = format_assistant_content(thinking, action_or_response)
            
            messages.append({
                'role': 'assistant',
                'content': content
            })
            
            # Add observation as tool message
            observation = step.get('observation')
            if observation:
                obs_text = observation_to_text(observation)
                if obs_text.strip():
                    messages.append({
                        'role': 'tool',
                        'content': obs_text
                    })
    
    # Validate: must have at least system + user + one assistant
    if len(messages) < 3:
        return None
    
    # Check that messages alternate properly
    # ms-swift expects: system, user, assistant, tool, assistant, tool, ...
    # We just need to ensure no consecutive assistant messages without tool in between
    # (except for the last one which may be assistant)
    
    return messages


def process_trial(trial_dir: Path) -> Optional[Dict[str, Any]]:
    """Process a single trial directory."""
    trajectory_path = trial_dir / 'agent' / 'trajectory.json'
    if not trajectory_path.exists():
        return None
    
    try:
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            trajectory = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  Warning: Failed to parse {trajectory_path}: {e}")
        return None
    
    messages = trajectory_to_messages(trajectory)
    if not messages:
        print(f"  Warning: No valid messages from {trial_dir.name}")
        return None
    
    return {'messages': messages}


def main():
    parser = argparse.ArgumentParser(description='Convert paper2arm trajectory data to ms-swift format')
    parser.add_argument('--input-dir', required=True, help='Input directory containing trial subdirectories')
    parser.add_argument('--output-file', required=True, help='Output JSONL file path')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    # Find all trial directories
    trial_dirs = []
    for name in sorted(os.listdir(input_dir)):
        trial_path = input_dir / name
        if trial_path.is_dir() and (trial_path / 'agent' / 'trajectory.json').exists():
            trial_dirs.append(trial_path)
    
    print(f"Found {len(trial_dirs)} trial directories")
    
    # Process each trial
    samples = []
    skipped = 0
    
    for i, trial_dir in enumerate(trial_dirs):
        if args.max_samples and len(samples) >= args.max_samples:
            break
        
        print(f"Processing {i+1}/{len(trial_dirs)}: {trial_dir.name}")
        sample = process_trial(trial_dir)
        if sample:
            samples.append(sample)
        else:
            skipped += 1
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nDone!")
    print(f"  Total trials: {len(trial_dirs)}")
    print(f"  Successful: {len(samples)}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {output_file}")
    
    # Print a sample for verification
    if samples:
        print("\n" + "="*60)
        print("Sample output (first sample, first 3 messages):")
        print("="*60)
        sample = samples[0]
        for msg in sample['messages'][:5]:
            content_preview = msg['content'][:200].replace('\n', '\\n')
            print(f"  {msg['role']}: {content_preview}...")


if __name__ == '__main__':
    main()
