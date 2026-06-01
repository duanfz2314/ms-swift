#!/usr/bin/env python3
"""Convert query-response JSON/JSONL datasets to ms-swift messages format.

Rules align with ms-swift ResponsePreprocessor / AlpacaPreprocessor:
- query-response: system + history + query/response -> messages
- alpaca: instruction + input -> user query, output -> assistant response

Other fields (images, videos, anchors, anchor_type, shape, ...) are kept as-is.

Example:
    python query_response_to_messages.py \
      --input-file train_query_response.jsonl \
      --output-file train_messages.jsonl
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SYSTEM_KEYS = ('system', 'system_prompt')
QUERY_KEYS = ('query', 'prompt', 'input', 'instruction', 'question', 'problem', 'text')
RESPONSE_KEYS = (
    'response', 'answer', 'output', 'targets', 'target', 'answer_key', 'answers', 'solution', 'completion', 'content',
)
# `text` / `content` may collide with query; prefer query keys first when resolving query.
QUERY_KEYS_ORDERED = ('query', 'prompt', 'input', 'instruction', 'question', 'problem')
RESPONSE_KEYS_ORDERED = ('response', 'answer', 'output', 'targets', 'target', 'answer_key', 'answers', 'solution',
                        'completion')
ALPACA_KEYS = ('instruction', 'input', 'output')
FIELDS_TO_DROP_AFTER_CONVERT = set(SYSTEM_KEYS + QUERY_KEYS + RESPONSE_KEYS + ('history', ) + ALPACA_KEYS)


def history_to_messages(history: List[List[Any]], system: Optional[str] = None) -> List[Dict[str, str]]:
    """Same contract as swift.template.utils.history_to_messages."""
    messages: List[Dict[str, str]] = []
    if system is not None:
        messages.append({'role': 'system', 'content': system})
    for turn in history:
        if not isinstance(turn, (list, tuple)) or len(turn) < 1:
            raise ValueError(f'Each history turn must be [query, response?], got: {turn!r}')
        query, response = (turn[0], turn[1] if len(turn) > 1 else None)
        if query is not None:
            messages.append({'role': 'user', 'content': str(query)})
        if response is not None:
            messages.append({'role': 'assistant', 'content': str(response)})
    return messages


def _pop_first(row: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row.pop(key)
    return None


def _pick_response(response: Any, *, random_pick: bool, seed: int) -> Any:
    if response is None:
        return None
    if isinstance(response, (list, tuple)):
        if not response:
            return None
        if random_pick:
            rng = random.Random(seed)
            return rng.choice(list(response))
        return response[0]
    return response


def _resolve_query(row: Dict[str, Any]) -> Any:
    instruction = row.get('instruction')
    input_ = row.get('input')
    if instruction is not None or input_ is not None:
        instruction = '' if instruction is None else str(instruction)
        input_ = '' if input_ is None else str(input_)
        if instruction and input_:
            return f'{instruction}\n{input_}'
        return instruction or input_ or None
    return _pop_first(row, QUERY_KEYS_ORDERED)


def _resolve_response(row: Dict[str, Any], *, random_pick: bool, seed: int) -> Any:
    if 'output' in row and 'response' not in row:
        return _pick_response(row.pop('output'), random_pick=random_pick, seed=seed)
    return _pick_response(_pop_first(row, RESPONSE_KEYS_ORDERED), random_pick=random_pick, seed=seed)


def convert_row(row: Dict[str, Any], *, infer_mode: bool = False, random_response: bool = False,
                seed: int = 42) -> Dict[str, Any]:
    """Convert one sample. If `messages` already exists, only strip legacy fields."""
    item = copy.deepcopy(row)
    if 'messages' in item:
        for key in FIELDS_TO_DROP_AFTER_CONVERT:
            item.pop(key, None)
        return item

    history = _pop_first(item, ('history', )) or []
    if isinstance(history, str):
        history = ast.literal_eval(history)
    if not isinstance(history, list):
        raise ValueError(f'`history` must be a list, got {type(history)}')

    system = _pop_first(item, SYSTEM_KEYS)
    query = _resolve_query(item)
    response = _resolve_response(item, random_pick=random_response, seed=seed)

    for key in list(item.keys()):
        if key in FIELDS_TO_DROP_AFTER_CONVERT:
            item.pop(key, None)

    if query is None and not history:
        raise ValueError('Sample has no `messages` and no `query`/`history`/`instruction` fields.')

    history = list(history)
    if query is not None:
        history.append([query, None if infer_mode else response])

    item['messages'] = history_to_messages(history, system)
    return item


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == '.jsonl':
        samples: List[Dict[str, Any]] = []
        with input_path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as e:
                    raise ValueError(f'Invalid JSONL at line {line_no}: {e}') from e
                if not isinstance(row, dict):
                    raise ValueError(f'Each JSONL row must be an object at line {line_no}')
                samples.append(row)
        return samples

    if suffix != '.json':
        raise ValueError(f'Unsupported input extension: {input_path.suffix}. Use .json or .jsonl')

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('data', 'samples', 'records', 'items'):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise ValueError(f'Invalid JSON top-level type: {type(data)}')


def save_samples(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == '.jsonl':
        with output_path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return
    if suffix != '.json':
        raise ValueError(f'Unsupported output extension: {output_path.suffix}. Use .json or .jsonl')
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert query-response JSON/JSONL to messages format.')
    parser.add_argument('--input-file', required=True, help='Input .json or .jsonl (query-response format).')
    parser.add_argument('--output-file', required=True, help='Output .json or .jsonl (messages format).')
    parser.add_argument(
        '--infer-mode',
        action='store_true',
        help='Omit the current-turn assistant response (for inference inputs only).')
    parser.add_argument(
        '--random-response',
        action='store_true',
        help='If response is a list, pick one at random (matches RANDOM_DATASET_RESPONSE).')
    parser.add_argument('--seed', type=int, default=42, help='Seed when --random-response is set.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_response:
        os.environ.setdefault('RANDOM_DATASET_RESPONSE', 'True')

    input_path = Path(args.input_file)
    samples = load_samples(input_path)
    converted = [
        convert_row(
            sample,
            infer_mode=args.infer_mode,
            random_response=args.random_response,
            seed=args.seed,
        ) for sample in samples
    ]
    save_samples(Path(args.output_file), converted)
    print(f'Converted {len(converted)} samples: {args.input_file} -> {args.output_file}')


if __name__ == '__main__':
    main()
