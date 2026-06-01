#!/usr/bin/env python3
"""Merge two JSON/JSONL files and shuffle the combined samples.

Supports:
- .json  (top-level list, or dict with data/samples/records/items)
- .jsonl (one JSON object per line)

Example (JSON array — recommended if you use .json everywhere):
    python merge_shuffle_json.py \
      --input-file a.json b.json \
      --output-file merged.json \
      --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

from json_dataset_io import load_dataset_file, save_dataset_file


def merge_and_shuffle(
    input_paths: List[Path],
    *,
    seed: int,
    shuffle: bool,
) -> Tuple[list, list]:
    merged = []
    counts = []
    for path in input_paths:
        part = load_dataset_file(path)
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
        default=2,
        help='Indent for .json output (default: 2). Use 0 for compact single-line array.')
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
    indent = args.indent if args.indent and args.indent > 0 else 0
    save_dataset_file(args.output_file, merged, indent=indent)

    parts = ', '.join(f'{path.name}={n}' for path, n in zip(input_paths, counts))
    action = 'merged' if args.no_shuffle else f'shuffled (seed={args.seed})'
    print(f'{action}: {parts} -> total={len(merged)} -> {args.output_file}')


if __name__ == '__main__':
    main()
