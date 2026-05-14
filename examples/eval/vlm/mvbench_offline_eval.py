#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline MVBench evaluation helper for swift eval + VLMEvalKit.

Use case:
1. You already have local MVBench raw files, typically:
   <root>/json/*.json
   <root>/video/...
2. The server is offline so VLMEvalKit cannot auto-download the dataset.

This script keeps code changes minimal by:
- generating ``MVBench.tsv`` from local JSON files;
- monkey-patching ``MVBench.prepare_dataset`` at runtime so VLMEvalKit uses the local root directly;
- calling ``swift eval`` pipeline via ``eval_main``.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


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


def _read_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_mvbench_tsv(mvbench_root: Path, allow_missing: bool = False) -> Tuple[Path, int]:
    json_dir = mvbench_root / 'json'
    video_root = mvbench_root / 'video'
    if not json_dir.is_dir():
        raise FileNotFoundError(f'json directory not found: {json_dir}')
    if not video_root.is_dir():
        raise FileNotFoundError(f'video directory not found: {video_root}')

    data_list: List[dict] = []
    missing_files: List[str] = []
    for task_type, (json_name, prefix_template, data_type, has_bound) in TYPE_DATA_LIST.items():
        json_file = json_dir / json_name
        if not json_file.exists():
            raise FileNotFoundError(f'Missing task annotation file: {json_file}')
        task_records = _read_json(json_file)
        prefix = prefix_template.replace('your_data_path', 'video')
        for item in task_records:
            video_rel = item['video']
            video_abs = mvbench_root / prefix / video_rel
            if not video_abs.exists():
                missing_files.append(str(video_abs))
                if allow_missing:
                    continue
                raise FileNotFoundError(
                    f'Missing video file: {video_abs}\n'
                    'Tip: MVBench has known NTURGB-D missing files in some mirrors. '
                    'If you want to continue with available samples, add --allow-missing.')
            data_list.append({
                'task_type': task_type,
                'prefix': prefix,
                'data_type': data_type,
                'bound': has_bound,
                'start': item.get('start'),
                'end': item.get('end'),
                'video': video_rel,
                'question': item['question'],
                'answer': item['answer'],
                'candidates': item['candidates'],
            })

    data_df = pd.DataFrame(data_list)
    data_df = data_df.assign(index=range(len(data_df)))
    tsv_path = mvbench_root / 'MVBench.tsv'
    data_df.to_csv(tsv_path, sep='\t', index=False)
    return tsv_path, len(missing_files)


def patch_mvbench_prepare_dataset(mvbench_root: Path, tsv_path: Path) -> None:
    from vlmeval.dataset.mvbench import MVBench
    from vlmeval.dataset.utils.mvbench import Stack, ToTorchFormatTensor
    import torchvision.transforms as T

    def _prepare_dataset(self, dataset_name='MVBench', repo_id='OpenGVLab/MVBench'):
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        # Keep the same defaults as upstream MVBench implementation.
        self.nframe = 8
        self.frame_fps = 3
        self.transform = T.Compose([Stack(), ToTorchFormatTensor()])
        return {'root': str(mvbench_root), 'data_file': str(tsv_path)}

    MVBench.prepare_dataset = _prepare_dataset


def main():
    parser = argparse.ArgumentParser(description='Run MVBench evaluation in offline mode with local raw dataset.')
    parser.add_argument('--mvbench_root', type=str, required=True, help='Local MVBench root containing json/ and video/.')
    parser.add_argument('--model', type=str, required=True, help='Model id/path for swift eval.')
    parser.add_argument('--infer_backend', type=str, default='vllm', help='Swift infer backend, e.g. vllm/transformers.')
    parser.add_argument('--eval_url', type=str, default=None, help='Existing OpenAI-format endpoint, optional.')
    parser.add_argument('--eval_limit', type=int, default=None, help='Sample limit for quick verification.')
    parser.add_argument('--eval_num_proc', type=int, default=4, help='Concurrent eval workers.')
    parser.add_argument('--eval_output_dir', type=str, default='eval_output', help='Directory for eval artifacts.')
    parser.add_argument('--allow_missing', action='store_true', help='Skip samples whose video files are missing.')
    args = parser.parse_args()

    mvbench_root = Path(args.mvbench_root).expanduser().resolve()
    tsv_path, missing_count = build_mvbench_tsv(mvbench_root, allow_missing=args.allow_missing)
    patch_mvbench_prepare_dataset(mvbench_root, tsv_path)

    # Ensure swift offline pre-check points to this local dataset root.
    os.environ.setdefault('LMU_DATA_DIR', str(mvbench_root))
    if args.eval_url:
        os.environ.setdefault('SWIFT_SKIP_VLMEVAL_DATA_CHECK', '1')

    from swift import EvalArguments, eval_main

    eval_args = EvalArguments(
        model=args.model,
        infer_backend=args.infer_backend,
        eval_backend='VLMEvalKit',
        eval_dataset=['MVBench'],
        eval_url=args.eval_url,
        eval_limit=args.eval_limit,
        eval_num_proc=args.eval_num_proc,
        eval_output_dir=args.eval_output_dir,
    )
    report = eval_main(eval_args)
    print('=== MVBench offline eval completed ===')
    print(f'Local TSV: {tsv_path}')
    if missing_count:
        print(f'Warning: skipped {missing_count} missing videos.')
    print(report)


if __name__ == '__main__':
    main()
