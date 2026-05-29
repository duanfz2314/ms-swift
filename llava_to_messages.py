#!/usr/bin/env python3
"""Convert LLaVA-style datasets to ms-swift messages JSON/JSONL (with anchor fields).

Supported inputs:
1. Classic LLaVA-Instruct (liuhaotian): list/json with id, image, conversations[{from, value}]
2. LLaVA pretrain-style: {image, text} or {image, caption}
3. Already converted rows with messages/images — only anchor fields are added

Defaults (for qwen3_vl_anchor training):
- anchor_type = 0  (no crop / no draw)
- anchors = [0, 0, 0, 0]

Example:
    python llava_to_messages.py \
      --input-file llava_instruct_150k.json \
      --output-file llava_messages.jsonl \
      --image-folder /data/llava_images
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

DEFAULT_ANCHORS = [0, 0, 0, 0]
DEFAULT_ANCHOR_TYPE = 0

HUMAN_ROLES = frozenset({'human', 'user'})
GPT_ROLES = frozenset({'gpt', 'chatgpt', 'assistant', 'bot'})


def load_llava_samples(input_path: Path) -> List[Dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == '.jsonl':
        samples: List[Dict[str, Any]] = []
        with input_path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if not isinstance(row, dict):
                    raise ValueError(f'JSONL line {line_no} must be an object')
                samples.append(row)
        return samples

    if suffix != '.json':
        raise ValueError(f'Unsupported input extension: {suffix}. Use .json or .jsonl')

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
        raise ValueError(f'Unsupported output extension: {suffix}. Use .json or .jsonl')
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def parse_media_roots(spec: Optional[str]) -> Dict[str, Path]:
    """Parse 'coco=/path/a,gqa=/path/b' into prefix -> root."""
    if not spec:
        return {}
    roots: Dict[str, Path] = {}
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f'Invalid --media-roots entry {part!r}, expected name=path')
        name, path = part.split('=', 1)
        roots[name.strip()] = Path(path.strip())
    return roots


def resolve_image_path(image: str, image_folder: Optional[Path], media_roots: Dict[str, Path]) -> str:
    image = image.replace('\\', '/')
    if os.path.isabs(image) and os.path.exists(image):
        return image

    if image_folder is not None:
        joined = image_folder / image
        if joined.exists():
            return str(joined.resolve())

    # LLaVA-Instruct path prefixes (same layout as ms-swift LLaVAInstructPreprocessor).
    prefix_rules = (
        ('coco/', 'coco'),
        ('gqa/', 'gqa'),
        ('ocr_vqa/', 'ocr_vqa'),
        ('textvqa/', 'textvqa'),
        ('VG_100K/', 'VG_100K'),
        ('VG_100K_2/', 'VG_100K_2'),
    )
    for prefix, key in prefix_rules:
        if prefix in image or image.startswith(prefix.rstrip('/') + '/'):
            root = media_roots.get(key)
            if root is None:
                continue
            rel = image
            if prefix == 'coco/':
                rel = image.replace('coco/', '')
            elif prefix == 'gqa/':
                rel = image.replace('gqa/', '')
            elif prefix == 'textvqa/':
                rel = image.replace('textvqa/', '')
            elif prefix in ('VG_100K/', 'VG_100K_2/'):
                rel = image.replace('vg/', '')
            candidate = root / rel
            if candidate.exists():
                return str(candidate.resolve())

    return str((image_folder / image).resolve()) if image_folder else image


def read_image_shape(image_path: str) -> Optional[List[int]]:
    if Image is None or not os.path.isfile(image_path):
        return None
    try:
        with Image.open(image_path) as img:
            w, h = img.size
        return [w, h]
    except Exception:
        return None


def _normalize_role(raw_role: str) -> str:
    role = raw_role.strip().lower()
    if role in HUMAN_ROLES:
        return 'user'
    if role in GPT_ROLES:
        return 'assistant'
    if role == 'system':
        return 'system'
    raise ValueError(f'Unknown conversation role: {raw_role!r}')


def conversations_to_messages(conversations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            raise ValueError(f'Each conversation turn must be a dict, got {type(turn)}')
        # ShareGPT-style in conversations: {human, assistant}
        if 'human' in turn or 'assistant' in turn:
            if turn.get('human') is not None:
                messages.append({'role': 'user', 'content': str(turn['human'])})
            if turn.get('assistant') is not None:
                messages.append({'role': 'assistant', 'content': str(turn['assistant'])})
            continue
        role_raw = turn.get('from') or turn.get('role')
        content = turn.get('value')
        if content is None:
            content = turn.get('content')
        if role_raw is None or content is None:
            raise ValueError(f'Invalid conversation turn (need from/role + value/content): {turn}')
        messages.append({'role': _normalize_role(str(role_raw)), 'content': str(content)})
    if not messages:
        raise ValueError('Empty conversations')
    return messages


def _extract_images_field(row: Dict[str, Any]) -> List[str]:
    images = row.get('images')
    if images is None:
        image = row.get('image')
        if image is None:
            return []
        if isinstance(image, list):
            return [str(x) for x in image]
        return [str(image)]
    if isinstance(images, str):
        return [images]
    if isinstance(images, list):
        out: List[str] = []
        for item in images:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                path = item.get('path') or item.get('file') or item.get('image')
                if path:
                    out.append(str(path))
            else:
                out.append(str(item))
        return out
    return [str(images)]


def _is_llava_conversation_row(row: Dict[str, Any]) -> bool:
    return 'conversations' in row and 'messages' not in row


def _is_pretrain_row(row: Dict[str, Any]) -> bool:
    return 'messages' not in row and 'conversations' not in row and row.get('image') and (
        row.get('text') is not None or row.get('caption') is not None)


def convert_llava_row(
    row: Dict[str, Any],
    *,
    image_folder: Optional[Path],
    media_roots: Dict[str, Path],
    anchors: List[int],
    anchor_type: int,
    with_shape: bool,
    keep_id: bool,
) -> Optional[Dict[str, Any]]:
    item: Dict[str, Any] = {}

    if keep_id and row.get('id') is not None:
        item['id'] = row['id']

    if 'messages' in row:
        item['messages'] = row['messages']
    elif _is_llava_conversation_row(row):
        item['messages'] = conversations_to_messages(row['conversations'])
    elif _is_pretrain_row(row):
        caption = row.get('text') if row.get('text') is not None else row.get('caption')
        item['messages'] = [
            {'role': 'user', 'content': '<image>\nDescribe this image.'},
            {'role': 'assistant', 'content': str(caption)},
        ]
    else:
        raise ValueError(
            'Unsupported sample: need `conversations`, `messages`, or pretrain fields (`image` + `text`/`caption`).')

    raw_images = _extract_images_field(row)
    if not raw_images:
        return None

    resolved = [resolve_image_path(p, image_folder, media_roots) for p in raw_images]
    item['images'] = resolved

    item['anchors'] = list(anchors)
    item['anchor_type'] = int(anchor_type)

    if with_shape and resolved:
        shape = read_image_shape(resolved[0])
        if shape is not None:
            item['shape'] = shape

    return item


def convert_dataset(
    samples: List[Dict[str, Any]],
    *,
    image_folder: Optional[Path],
    media_roots: Dict[str, Path],
    anchors: List[int],
    anchor_type: int,
    with_shape: bool,
    keep_id: bool,
    skip_missing_image: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    converted: List[Dict[str, Any]] = []
    skipped = 0
    for idx, row in enumerate(samples):
        if not isinstance(row, dict):
            raise ValueError(f'Sample at index {idx} must be a dict, got {type(row)}')
        try:
            out = convert_llava_row(
                row,
                image_folder=image_folder,
                media_roots=media_roots,
                anchors=anchors,
                anchor_type=anchor_type,
                with_shape=with_shape,
                keep_id=keep_id,
            )
        except ValueError as e:
            raise ValueError(f'Sample index {idx}: {e}') from e
        if out is None:
            skipped += 1
            continue
        if skip_missing_image and out['images'] and not os.path.isfile(out['images'][0]):
            skipped += 1
            continue
        converted.append(out)
    return converted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert LLaVA dataset JSON to ms-swift messages + anchor fields.')
    parser.add_argument('--input-file', required=True, help='LLaVA .json (list) or .jsonl')
    parser.add_argument('--output-file', required=True, help='Output .json or .jsonl')
    parser.add_argument(
        '--image-folder',
        type=str,
        default=None,
        help='Root folder prepended to `image` paths (LLaVA --image_folder).')
    parser.add_argument(
        '--media-roots',
        type=str,
        default=None,
        help='Optional comma-separated roots, e.g. coco=/data/coco,gqa=/data/gqa')
    parser.add_argument(
        '--anchors',
        type=int,
        nargs=4,
        default=DEFAULT_ANCHORS,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='Anchor box (default: 0 0 0 0).')
    parser.add_argument(
        '--anchor-type',
        type=int,
        default=DEFAULT_ANCHOR_TYPE,
        choices=[0, 1, 2],
        help='0=none (default), 1=crop, 2=draw.')
    parser.add_argument('--with-shape', action='store_true', help='Read image width/height into `shape` [w,h].')
    parser.add_argument('--keep-id', action='store_true', help='Keep original `id` field in output.')
    parser.add_argument(
        '--skip-missing-image',
        action='store_true',
        help='Skip samples whose resolved image path does not exist.')
    parser.add_argument('--limit', type=int, default=None, help='Convert at most N samples (debug).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    samples = load_llava_samples(input_path)
    if args.limit is not None:
        samples = samples[:max(0, args.limit)]

    image_folder = Path(args.image_folder).resolve() if args.image_folder else None
    media_roots = parse_media_roots(args.media_roots)

    converted, skipped = convert_dataset(
        samples,
        image_folder=image_folder,
        media_roots=media_roots,
        anchors=list(args.anchors),
        anchor_type=args.anchor_type,
        with_shape=args.with_shape,
        keep_id=args.keep_id,
        skip_missing_image=args.skip_missing_image,
    )
    save_samples(Path(args.output_file), converted)
    print(f'Converted {len(converted)} samples (skipped {skipped}) -> {args.output_file}')


if __name__ == '__main__':
    main()
