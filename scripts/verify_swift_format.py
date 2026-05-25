#!/usr/bin/env python3
"""
Verify that the converted paper2arm data is compatible with ms-swift training.

Checks:
1. JSONL format validity
2. Message roles are valid
3. No consecutive assistant messages (would cause template errors)
4. Assistant messages contain <think> tags for Qwen3.5/3.6
5. Tool call/response sequence shape
6. Token length statistics (approximate)
7. Loss mask explanation for ms-swift swift backend
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def validate_sample(sample: dict, idx: int, expect_mask_tool_calls: bool = False) -> list[str]:
    """Validate a single sample and return list of issues."""
    issues = []
    msgs = sample.get('messages', [])
    
    if not msgs:
        issues.append("Empty messages")
        return issues
    
    valid_roles = {'system', 'user', 'assistant', 'tool', 'tool_call', 'tool_response'}
    
    # Check roles
    for j, msg in enumerate(msgs):
        role = msg.get('role')
        if role not in valid_roles:
            issues.append(f"Invalid role '{role}' at message {j}")
        if 'content' not in msg:
            issues.append(f"Missing content at message {j}")
    
    # Check no consecutive assistant messages
    for j in range(1, len(msgs)):
        if msgs[j-1]['role'] == 'assistant' and msgs[j]['role'] == 'assistant':
            issues.append(f"Consecutive assistant messages at {j-1},{j}")
    
    # Check tool messages follow assistant/tool_call shape.
    for j in range(len(msgs)):
        if msgs[j]['role'] == 'tool':
            if j == 0 or msgs[j-1]['role'] != 'assistant':
                issues.append(f"Tool message at {j} not preceded by assistant")
        if msgs[j]['role'] == 'tool_call':
            if j == 0 or msgs[j-1]['role'] not in {'assistant', 'tool_call'}:
                issues.append(f"Tool call at {j} not preceded by assistant/tool_call")
            try:
                parsed = json.loads(msgs[j].get('content') or '')
                if not isinstance(parsed, dict) or 'name' not in parsed or 'arguments' not in parsed:
                    issues.append(f"Tool call at {j} is not a {{name, arguments}} object")
            except json.JSONDecodeError:
                issues.append(f"Tool call at {j} content is not JSON")
            if expect_mask_tool_calls and msgs[j].get('loss') is not False:
                issues.append(f"Tool call at {j} should set loss=false")
        if msgs[j]['role'] == 'tool_response':
            if j == 0 or msgs[j-1]['role'] not in {'tool_call', 'tool_response'}:
                issues.append(f"Tool response at {j} not preceded by tool_call/tool_response")
    
    # Check assistant messages have <think> for Qwen3.5/3.6
    for j, msg in enumerate(msgs):
        if msg['role'] == 'assistant':
            if '<think>' not in msg['content']:
                issues.append(f"Assistant message {j} missing <think> tag")
            if '</think>' not in msg['content']:
                issues.append(f"Assistant message {j} missing </think> tag")
    
    # Check structure: system, user, [assistant, tool]*, [assistant]
    if len(msgs) >= 1 and msgs[0]['role'] != 'system':
        issues.append("First message should be system")
    if len(msgs) >= 2 and msgs[1]['role'] != 'user':
        issues.append("Second message should be user")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description='Verify ms-swift training data format')
    parser.add_argument('--input-file', required=True, help='Input JSONL file')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to check')
    parser.add_argument(
        '--expect-mask-tool-calls',
        action='store_true',
        help='Require every tool_call message to set loss=false.',
    )
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if args.max_samples and len(samples) >= args.max_samples:
                break
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples")
    print("=" * 60)
    
    # Validate each sample
    total_issues = 0
    role_counts = Counter()
    total_chars = 0
    sample_msg_counts = []
    
    for i, sample in enumerate(samples):
        issues = validate_sample(sample, i, expect_mask_tool_calls=args.expect_mask_tool_calls)
        if issues:
            total_issues += len(issues)
            print(f"Sample {i} issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        msgs = sample['messages']
        sample_msg_counts.append(len(msgs))
        total_chars += sum(len(m.get('content', '')) for m in msgs)
        for msg in msgs:
            role_counts[msg['role']] += 1
    
    print("=" * 60)
    print("Summary:")
    print(f"  Total samples: {len(samples)}")
    print(
        "  Samples with issues: "
        f"{sum(1 for i in range(len(samples)) if validate_sample(samples[i], i, args.expect_mask_tool_calls))}"
    )
    print(f"  Total issues: {total_issues}")
    print()
    print("  Message counts:")
    for role, count in sorted(role_counts.items()):
        print(f"    {role:12s}: {count}")
    print()
    print("  Sample statistics:")
    print(f"    Avg messages/sample: {sum(sample_msg_counts)/len(sample_msg_counts):.1f}")
    print(f"    Min messages: {min(sample_msg_counts)}")
    print(f"    Max messages: {max(sample_msg_counts)}")
    print(f"    Avg chars/message: {total_chars/sum(sample_msg_counts):.0f}")
    print(f"    Total data size: {total_chars / 1024 / 1024:.2f} MB")
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 chars for English)
    estimated_tokens = total_chars / 4
    print(f"    Estimated total tokens: {estimated_tokens:,.0f}")
    print(f"    Estimated tokens/sample: {estimated_tokens/len(samples):,.0f}")
    
    print()
    print("=" * 60)
    print("Loss mask analysis (ms-swift swift backend, --loss_scale default):")
    print("  - system messages: MASKED (no loss)")
    print("  - user messages: MASKED (no loss)")
    print("  - assistant messages: UNMASKED (compute loss)")
    print("  - tool/tool_response messages: MASKED (environment feedback, no loss)")
    print("  - tool_call messages: rendered as assistant tool-call text and UNMASKED by default")
    print("    Set loss=false on tool_call messages to force MASKED labels for tool calls.")
    print()
    assistant_chars = sum(
        len(m['content']) for s in samples for m in s['messages'] if m['role'] == 'assistant'
    )
    tool_call_chars = sum(
        len(m['content']) for s in samples for m in s['messages'] if m['role'] == 'tool_call'
    )
    masked_tool_call_chars = sum(
        len(m['content'])
        for s in samples for m in s['messages']
        if m['role'] == 'tool_call' and m.get('loss') is False
    )
    total_chars_all = sum(
        len(m['content']) for s in samples for m in s['messages']
    )
    default_train_chars = assistant_chars + tool_call_chars
    explicit_train_chars = assistant_chars + (tool_call_chars - masked_tool_call_chars)
    print(f"  Assistant content: {assistant_chars / 1024:.0f} KB ({assistant_chars/total_chars_all*100:.1f}% of total)")
    print(f"  Tool-call JSON payloads: {tool_call_chars / 1024:.0f} KB ({tool_call_chars/total_chars_all*100:.1f}% of total)")
    print(f"  Tool-call payloads with loss=false: {masked_tool_call_chars / 1024:.0f} KB")
    print(f"  Default trainable character proxy: {default_train_chars / 1024:.0f} KB ({default_train_chars/total_chars_all*100:.1f}% of total)")
    print(f"  Explicit-loss trainable proxy: {explicit_train_chars / 1024:.0f} KB ({explicit_train_chars/total_chars_all*100:.1f}% of total)")
    
    if total_issues == 0:
        print("\n✅ All samples passed validation!")
    else:
        print(f"\n⚠️  Found {total_issues} issues across {len(samples)} samples")


if __name__ == '__main__':
    main()
