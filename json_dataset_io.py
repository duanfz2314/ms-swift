#!/usr/bin/env python3
"""Load/save ms-swift style datasets as JSON array (.json) or JSONL (.jsonl).

- `.json`  -> one file, top-level list: [{...}, {...}]
- `.jsonl` -> one JSON object per line

Both are valid for `swift sft --dataset your_file.json`.
Use this module in helper scripts so JSON and JSONL behave the same.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, Path]


def load_dataset_file(path: PathLike) -> List[Dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        rows: List[Dict[str, Any]] = []
        with path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f'{path}: invalid JSONL at line {line_no}: {e}\n'
                        f'  hint: .jsonl must be one complete {{...}} per line, not a pretty-printed array.',
                    ) from e
                if not isinstance(row, dict):
                    raise ValueError(f'{path}: line {line_no} must be a JSON object')
                rows.append(row)
        return rows

    if suffix != '.json':
        raise ValueError(f'{path}: unsupported extension {suffix!r}. Use .json or .jsonl')

    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f'{path}: item {idx} must be an object')
        return data

    if isinstance(data, dict):
        for key in ('data', 'samples', 'records', 'items'):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]

    raise ValueError(f'{path}: JSON root must be a list or dict, got {type(data)}')


def save_dataset_file(path: PathLike, rows: List[Dict[str, Any]], *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == '.jsonl':
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return

    if suffix != '.json':
        raise ValueError(f'{path}: unsupported extension {suffix!r}. Use .json or .jsonl')

    with path.open('w', encoding='utf-8') as f:
        if indent and indent > 0:
            json.dump(rows, f, ensure_ascii=False, indent=indent)
        else:
            json.dump(rows, f, ensure_ascii=False, separators=(',', ':'))
