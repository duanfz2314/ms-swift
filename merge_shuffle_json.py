#!/usr/bin/env python3
"""Merge two JSON/JSONL files and shuffle the combined samples.

Supports:
- .json  (top-level list, or dict with data/samples/records/items)
- .jsonl (one JSON object per line)

Example:
    python merge_shuffle_json.py \
      --input-file a.jsonl b.jsonl \
      --output-file merged.jsonl \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
                    raise ValueError(f'{input_path}: invalid JSONL at line {line_no}: {e}') from e
                if not isinstance(row, dict):
                    raise ValueError(f'{input_path}: line {line_no} must be a JSON object')
                samples.append(row)
        return samples

    if suffix != '.json':
        raise ValueError(f'{input_path}: unsupported extension {suffix}. Use .json or .jsonl')

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        for key in ('data', 'samples', 'records', 'items'):
            if isinstance(data.get(key), list):
                samples = data[key]
                break
        else:
            samples = [data]
    else:
        raise ValueError(f'{input_path}: invalid JSON top-level type {type(data)}')

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f'{input_path}: sample at index {idx} must be an object')
    return samples


def save_samples(output_path: Path, rows: List[Dict[str, Any]], *, indent: Optional[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == '.jsonl':
        with output_path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return
    if suffix != '.json':
        raise ValueError(f'Unsupported output extension: {suffix}. Use .json or .jsonl')
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=indent)


def merge_and_shuffle(
    input_paths: List[Path],
    *,
    seed: int,
    shuffle: bool,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    merged: List[Dict[str, Any]] = []
    counts: List[int] = []
    for path in input_paths:
        part = load_samples(path)
        counts.append(len(part))
        merged.extend(part)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(merged)
    return merged, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merge two or more JSON/JSONL files and shuffle.')
    parser.add_argument(
        '--input-file',
        nargs='+',
        required=True,
        help='Two or more input .json / .jsonl files.')
    parser.add_argument('--output-file', required=True, help='Output .json or .jsonl path.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling (default: 42).')
    parser.add_argument('--no-shuffle', action='store_true', help='Only merge, do not shuffle.')
    parser.add_argument(
        '--indent',
        type=int,
        default=None,
        help='Indent for .json output (default: compact list; use 2 for pretty print).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.input_file) < 2:
        raise SystemExit('Please provide at least two --input-file paths.')

    input_paths = [Path(p) for p in args.input_file]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f'Input file not found: {path}')

    merged, counts = merge_and_shuffle(input_paths, seed=args.seed, shuffle=not args.no_shuffle)
    save_samples(Path(args.output_file), merged, indent=args.indent)

    parts = ', '.join(f'{path.name}={n}' for path, n in zip(input_paths, counts))
    action = 'merged' if args.no_shuffle else f'shuffled (seed={args.seed})'
    print(f'{action}: {parts} -> total={len(merged)} -> {args.output_file}')


if __name__ == '__main__':
    main()
