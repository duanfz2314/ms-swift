#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Prepare local MVBench dataset for offline VLMEvalKit evaluation.

This script converts official MVBench local files (json/ + video/) into
MVBench.tsv with the same schema used by VLMEvalKit, and can optionally
register the dataset into HuggingFace cache layout so `swift eval` can
discover it without downloading.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


EXPECTED_MVBENCH_MD5 = 'fd21d36522cdedd46d84dc46715ad832'

# Aligned with open-compass/VLMEvalKit: vlmeval/dataset/mvbench.py
TYPE_DATA_LIST: Dict[str, Tuple[str, str, str, bool]] = {
    'Action Sequence': ('action_sequence.json', 'your_data_path/star/Charades_v1_480/', 'video', True),
    'Action Prediction': ('action_prediction.json', 'your_data_path/star/Charades_v1_480/', 'video', True),
    'Action Antonym': ('action_antonym.json', 'your_data_path/ssv2_video/', 'video', False),
    'Fine-grained Action': ('fine_grained_action.json', 'your_data_path/Moments_in_Time_Raw/videos/', 'video', False),
    'Unexpected Action': ('unexpected_action.json', 'your_data_path/FunQA_test/test/', 'video', False),
    'Object Existence': ('object_existence.json', 'your_data_path/clevrer/video_validation/', 'video', False),
    'Object Interaction': ('object_interaction.json', 'your_data_path/star/Charades_v1_480/', 'video', True),
    'Object Shuffle': ('object_shuffle.json', 'your_data_path/perception/videos/', 'video', False),
    'Moving Direction': ('moving_direction.json', 'your_data_path/clevrer/video_validation/', 'video', False),
    'Action Localization': ('action_localization.json', 'your_data_path/sta/sta_video/', 'video', True),
    'Scene Transition': ('scene_transition.json', 'your_data_path/scene_qa/video/', 'video', False),
    'Action Count': ('action_count.json', 'your_data_path/perception/videos/', 'video', False),
    'Moving Count': ('moving_count.json', 'your_data_path/clevrer/video_validation/', 'video', False),
    'Moving Attribute': ('moving_attribute.json', 'your_data_path/clevrer/video_validation/', 'video', False),
    'State Change': ('state_change.json', 'your_data_path/perception/videos/', 'video', False),
    'Fine-grained Pose': ('fine_grained_pose.json', 'your_data_path/nturgbd/', 'video', False),
    'Character Order': ('character_order.json', 'your_data_path/perception/videos/', 'video', False),
    'Egocentric Navigation': ('egocentric_navigation.json', 'your_data_path/vlnqa/', 'video', False),
    'Episodic Reasoning': ('episodic_reasoning.json', 'your_data_path/tvqa/frames_fps3_hq/', 'frame', True),
    'Counterfactual Inference': ('counterfactual_inference.json', 'your_data_path/clevrer/video_validation/', 'video',
                                 False),
}

COLUMN_ORDER = [
    'task_type',
    'prefix',
    'data_type',
    'bound',
    'start',
    'end',
    'video',
    'question',
    'answer',
    'candidates',
    'index',
]


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_hf_cache_root() -> Path:
    for env_name in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_HOME'):
        value = os.environ.get(env_name)
        if not value:
            continue
        root = Path(value)
        return root if root.name == 'hub' else (root / 'hub')
    return Path.home() / '.cache' / 'huggingface' / 'hub'


def build_mvbench_rows(root_dir: Path, allow_missing: bool) -> Tuple[List[dict], List[str]]:
    json_dir = root_dir / 'json'
    rows: List[dict] = []
    missing: List[str] = []

    for task_type, (json_name, prefix_tmpl, data_type, has_bound) in TYPE_DATA_LIST.items():
        json_file = json_dir / json_name
        if not json_file.exists():
            raise FileNotFoundError(f'JSON file not found: {json_file}')

        with json_file.open('r', encoding='utf-8') as f:
            samples = json.load(f)

        prefix = prefix_tmpl.replace('your_data_path', 'video')
        for sample in samples:
            rel_video = sample['video']
            abs_video = root_dir / prefix / rel_video
            if not abs_video.exists():
                missing.append(str(abs_video))
                if not allow_missing:
                    continue

            rows.append({
                'task_type': task_type,
                'prefix': prefix,
                'data_type': data_type,
                'bound': has_bound,
                'start': sample['start'] if 'start' in sample else None,
                'end': sample['end'] if 'end' in sample else None,
                'video': rel_video,
                'question': sample['question'],
                'answer': sample['answer'],
                'candidates': sample['candidates'],
            })

    return rows, missing


def write_mvbench_tsv(root_dir: Path, output_tsv: Path, allow_missing: bool) -> Tuple[int, int, str]:
    rows, missing = build_mvbench_rows(root_dir, allow_missing=allow_missing)
    if missing and not allow_missing:
        preview = '\n'.join(missing[:10])
        raise FileNotFoundError(
            f'Found {len(missing)} missing videos/frames. Example paths:\n{preview}\n'
            'Re-run with --allow-missing to generate subset TSV for debugging.'
        )

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_ORDER, delimiter='\t')
        writer.writeheader()
        for idx, row in enumerate(rows):
            row = dict(row)
            row['index'] = idx
            writer.writerow(row)
    return len(rows), len(missing), file_md5(output_tsv)


def register_to_hf_cache(dataset_root: Path, repo_id: str, branch: str) -> Path:
    org, name = repo_id.split('/', 1)
    cache_root = get_hf_cache_root()
    repo_root = cache_root / f'datasets--{org}--{name}'
    refs_dir = repo_root / 'refs'
    snapshots_dir = repo_root / 'snapshots'

    refs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshot_id = hashlib.sha1(str(dataset_root.resolve()).encode('utf-8')).hexdigest()
    snapshot_path = snapshots_dir / snapshot_id
    if snapshot_path.exists() or snapshot_path.is_symlink():
        snapshot_path.unlink()
    snapshot_path.symlink_to(dataset_root.resolve(), target_is_directory=True)

    (refs_dir / branch).write_text(f'{snapshot_id}\n', encoding='utf-8')
    return snapshot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate MVBench.tsv from local json/video folders and optionally register offline cache.')
    parser.add_argument('--mvbench-root', type=Path, required=True, help='Local MVBench root (must contain json/ and video/).')
    parser.add_argument('--output-tsv', type=Path, default=None, help='Output TSV path. Default: <mvbench-root>/MVBench.tsv')
    parser.add_argument(
        '--allow-missing',
        action='store_true',
        help='Allow missing videos and generate partial TSV (only for debugging, score not comparable).')
    parser.add_argument(
        '--register-cache',
        action='store_true',
        help='Register local path into HuggingFace cache layout for offline VLMEvalKit auto-discovery.')
    parser.add_argument(
        '--repo-id',
        default='OpenGVLab/MVBench',
        help='Dataset repo id used by VLMEvalKit cache lookup. Default: OpenGVLab/MVBench')
    parser.add_argument('--branch', default='main', help='Branch name used by cache refs. Default: main')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.mvbench_root.resolve()
    if not (root / 'json').exists() or not (root / 'video').exists():
        raise FileNotFoundError(f'Expect both json/ and video/ under: {root}')

    output_tsv = args.output_tsv.resolve() if args.output_tsv else (root / 'MVBench.tsv')
    rows, missing_cnt, generated_md5 = write_mvbench_tsv(root, output_tsv, allow_missing=args.allow_missing)

    print(f'[OK] TSV generated: {output_tsv}')
    print(f'[INFO] rows={rows}, missing={missing_cnt}, md5={generated_md5}')
    if generated_md5 != EXPECTED_MVBENCH_MD5:
        print('[WARN] MD5 mismatch with official MVBench.tsv. '
              'Offline swift eval may try to download dataset if strict integrity check is enabled.')
    else:
        print('[OK] MD5 matches official MVBench.tsv.')

    if args.register_cache:
        snapshot_path = register_to_hf_cache(root, repo_id=args.repo_id, branch=args.branch)
        print(f'[OK] Registered HF cache snapshot: {snapshot_path}')
        print('[TIP] Keep default HF cache env (or set HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE/HF_HOME accordingly).')

    print('\nSuggested eval command:')
    print('swift eval --eval_backend VLMEvalKit --eval_dataset MVBench '
          '--model <your_model> --infer_backend <your_backend> --eval_limit <N>')


if __name__ == '__main__':
    main()
