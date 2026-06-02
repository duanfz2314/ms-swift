#!/usr/bin/env python3
"""Merge single-turn dialogue JSON/JSONL into random multi-turn (1--3 rounds) interleaved samples.

Designed for video SFT (e.g. qwen3_vl_anchor with message-order anchors). Each input row is
one user/assistant turn; output rows concatenate 1--3 consecutive (after shuffle) rows into
one conversation. When every group has 3 rounds, output count is ~1/3 of input.

Example:
    python scripts/single_to_interleaved_dialogue.py \\
        --input data/single_turn.jsonl \\
        --output data/interleaved.jsonl \\
        --seed 42

Input (single-turn, messages form):
    {"messages": [{"role": "user", "content": "<video>Q1"}, {"role": "assistant", "content": "A1"}],
     "videos": ["a.mp4"], "anchors": [[1,2,3,4]], "anchor_type": 1}

Output (2-round interleaved):
    {"messages": [
        {"role": "user", "content": "<video>Q1"}, {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "<video>Q2"}, {"role": "assistant", "content": "A2"}],
     "videos": ["a.mp4", "b.mp4"],
     "anchors": [[1,2,3,4], [5,6,7,8]],
     "anchor_type": [1, 1]}
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

JsonDict = Dict[str, Any]

MEDIA_LIST_KEYS = ('videos', 'images', 'audios')
MEDIA_TAG = {'videos': '<video>', 'images': '<image>', 'audios': '<audio>'}
ANCHOR_KEYS = ('anchors', 'anchor')
ANCHOR_TYPE_KEYS = ('anchor_type',)


def _is_flat_xyxy_box(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return False
    if isinstance(value[0], (list, tuple, dict)):
        return False
    try:
        for i in range(4):
            float(value[i])
    except Exception:
        return False
    return True


def _load_records(path: Path) -> List[JsonDict]:
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return []
    if path.suffix.lower() == '.jsonl':
        records = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{line_no}: invalid JSON: {e}') from e
        return records
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f'{path}: root must be object or array')


def _write_jsonl(path: Path, records: Sequence[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _normalize_single_turn(record: JsonDict) -> JsonDict:
    """Convert query/response (+ history) to messages-style sample."""
    if 'messages' in record:
        out = copy.deepcopy(record)
    elif 'query' in record and 'response' in record:
        messages: List[JsonDict] = []
        system = record.get('system')
        if system:
            messages.append({'role': 'system', 'content': system})
        for turn in record.get('history') or []:
            if not isinstance(turn, (list, tuple)) or len(turn) < 2:
                continue
            messages.append({'role': 'user', 'content': turn[0]})
            messages.append({'role': 'assistant', 'content': turn[1]})
        messages.append({'role': 'user', 'content': record['query']})
        messages.append({'role': 'assistant', 'content': record['response']})
        out = {k: v for k, v in record.items() if k not in {'query', 'response', 'history', 'system'}}
        out['messages'] = messages
    elif 'conversations' in record:
        messages = []
        for turn in record['conversations']:
            role = turn.get('from', turn.get('role', 'user'))
            if role == 'human':
                role = 'user'
            elif role == 'gpt':
                role = 'assistant'
            content = turn.get('value', turn.get('content', ''))
            messages.append({'role': role, 'content': content})
        out = {k: v for k, v in record.items() if k != 'conversations'}
        out['messages'] = messages
    else:
        raise ValueError('Each record needs "messages", or "query"+"response", or "conversations"')

    msgs = out.get('messages') or []
    if not msgs:
        raise ValueError('Empty messages')
    roles = [m.get('role') for m in msgs if isinstance(m, dict)]
    if roles.count('assistant') < 1 or roles.count('user') < 1:
        raise ValueError(f'Sample is not a valid single-turn dialogue: roles={roles}')
    return out


def _count_rounds(messages: List[JsonDict]) -> int:
    """Count user turns (one round = user [+ assistant])."""
    return sum(1 for m in messages if isinstance(m, dict) and m.get('role') == 'user')


def _strip_leading_system(messages: List[JsonDict]) -> Tuple[Optional[JsonDict], List[JsonDict]]:
    if messages and messages[0].get('role') == 'system':
        return messages[0], messages[1:]
    return None, messages


def _merge_list_field(samples: Sequence[JsonDict], key: str) -> Optional[List[Any]]:
    merged: List[Any] = []
    for s in samples:
        val = s.get(key)
        if val is None:
            continue
        if not isinstance(val, list):
            merged.append(val)
            continue
        if key in ANCHOR_KEYS and _is_flat_xyxy_box(val):
            merged.append(list(val))
        else:
            merged.extend(val)
    return merged if merged else None


def _merge_anchor_types(samples: Sequence[JsonDict]) -> Any:
    types: List[Any] = []
    for s in samples:
        for k in ANCHOR_TYPE_KEYS:
            if k in s:
                types.append(s[k])
                break
    if not types:
        return None
    flat: List[Any] = []
    for t in types:
        if isinstance(t, (list, tuple)):
            flat.extend(t)
        else:
            flat.append(t)
    if len(flat) == 1:
        return flat[0]
    if all(x == flat[0] for x in flat):
        return flat[0]
    return flat


def _ensure_media_tags(messages: List[JsonDict], media_key: str, count: int) -> None:
    if count <= 0:
        return
    tag = MEDIA_TAG[media_key]
    user_indices = [i for i, m in enumerate(messages) if m.get('role') == 'user']
    if len(user_indices) < count:
        return
    # Assign tags to first `count` user turns that lack the tag.
    tagged = 0
    for idx in user_indices:
        if tagged >= count:
            break
        content = messages[idx].get('content', '')
        if not isinstance(content, str):
            continue
        if tag not in content:
            messages[idx] = dict(messages[idx])
            messages[idx]['content'] = f'{tag}{content}'
        tagged += 1


def merge_samples(samples: Sequence[JsonDict], *, ensure_tags: bool = True) -> JsonDict:
    if not samples:
        raise ValueError('empty samples')
    if len(samples) > 3:
        raise ValueError('at most 3 single-turn samples per merged conversation')

    normalized = [_normalize_single_turn(copy.deepcopy(s)) for s in samples]

    system_msg: Optional[JsonDict] = None
    merged_messages: List[JsonDict] = []

    for i, sample in enumerate(normalized):
        sys_part, body = _strip_leading_system(sample['messages'])
        if i == 0:
            system_msg = sys_part
            merged_messages.extend(body)
        else:
            if sys_part is not None:
                pass  # drop duplicate system in later samples
            merged_messages.extend(body)

    if system_msg is not None:
        merged_messages = [system_msg] + merged_messages

    merged: JsonDict = {}
    # Keep non-message metadata from first sample, then overlay merged media fields.
    skip_keys = {
        'messages', 'query', 'response', 'history', 'conversations', 'system',
        *MEDIA_LIST_KEYS, *ANCHOR_KEYS, *ANCHOR_TYPE_KEYS,
    }
    for k, v in normalized[0].items():
        if k not in skip_keys:
            merged[k] = copy.deepcopy(v)

    merged['messages'] = merged_messages

    for media_key in MEDIA_LIST_KEYS:
        vals = _merge_list_field(normalized, media_key)
        if vals is not None:
            merged[media_key] = vals
            if ensure_tags:
                _ensure_media_tags(merged['messages'], media_key, len(vals))

    anchors = _merge_list_field(normalized, 'anchors')
    if anchors is None:
        anchors = _merge_list_field(normalized, 'anchor')
        if anchors is not None:
            merged['anchors'] = anchors
    else:
        merged['anchors'] = anchors

    anchor_type = _merge_anchor_types(normalized)
    if anchor_type is not None:
        merged['anchor_type'] = anchor_type

    # Hint for qwen3_vl_anchor plugin (message-order is default).
    merged.setdefault('anchors_align', 'message_order')

    return merged


def _random_group_sizes(n: int, rng: random.Random, min_rounds: int, max_rounds: int) -> List[int]:
    sizes: List[int] = []
    i = 0
    while i < n:
        remaining = n - i
        upper = min(max_rounds, remaining)
        lower = min(min_rounds, upper)
        size = rng.randint(lower, upper)
        sizes.append(size)
        i += size
    return sizes


def build_interleaved_dataset(
    records: Sequence[JsonDict],
    *,
    seed: int = 42,
    min_rounds: int = 1,
    max_rounds: int = 3,
    shuffle: bool = True,
    ensure_tags: bool = True,
) -> List[JsonDict]:
    if min_rounds < 1 or max_rounds > 3 or min_rounds > max_rounds:
        raise ValueError('rounds must satisfy 1 <= min_rounds <= max_rounds <= 3')

    items = list(records)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(items)

    sizes = _random_group_sizes(len(items), rng, min_rounds, max_rounds)
    out: List[JsonDict] = []
    cursor = 0
    for size in sizes:
        chunk = items[cursor:cursor + size]
        cursor += size
        out.append(merge_samples(chunk, ensure_tags=ensure_tags))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', '-i', type=Path, required=True, help='Input .json or .jsonl (single-turn rows)')
    p.add_argument('--output', '-o', type=Path, required=True, help='Output .jsonl path')
    p.add_argument('--seed', type=int, default=42, help='Random seed for shuffle and group sizes')
    p.add_argument('--min-rounds', type=int, default=1, help='Min single-turn rows merged per output sample')
    p.add_argument('--max-rounds', type=int, default=3, help='Max single-turn rows merged per output sample')
    p.add_argument('--no-shuffle', action='store_true', help='Keep input order (still random group sizes)')
    p.add_argument('--no-media-tags', action='store_true', help='Do not auto-prepend <video>/<image> tags')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    records = _load_records(args.input)
    if not records:
        raise SystemExit(f'No records in {args.input}')

    merged = build_interleaved_dataset(
        records,
        seed=args.seed,
        min_rounds=args.min_rounds,
        max_rounds=args.max_rounds,
        shuffle=not args.no_shuffle,
        ensure_tags=not args.no_media_tags,
    )

    _write_jsonl(args.output, merged)

    in_n = len(records)
    out_n = len(merged)
    ratio = out_n / in_n if in_n else 0
    print(f'Input samples:  {in_n}')
    print(f'Output samples: {out_n} ({ratio:.2%} of input, ~1/{in_n/out_n:.1f} if >0)')
    print(f'Written: {args.output}')


if __name__ == '__main__':
    main()
