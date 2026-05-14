#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
离线环境下准备 MVBench 数据集，使 VLMEvalKit 无需联网即可加载。

原理：
  VLMEvalKit 的 MVBench 数据加载流程为：
    1. 通过 get_cache_path() 查找 HuggingFace 缓存目录
    2. 检查缓存中是否存在 MVBench.tsv 且数据完整 (check_integrity)
    3. 若完整则直接使用，否则尝试从 HuggingFace 下载（离线环境会失败）

  本脚本所做的工作：
    1. 解压 video/ 目录下的 zip 文件（如有）
    2. 移动 video/data0613/ 下的文件到正确位置（如有）
    3. 从 json/ 目录生成 MVBench.tsv（与 VLMEvalKit 内部逻辑完全一致）
    4. 在 HuggingFace 缓存目录下创建符合规范的缓存结构，指向本地数据

使用方法：
  python prepare_mvbench_offline.py --data-dir /path/to/your/MVBench

  其中 /path/to/your/MVBench 的目录结构应为：
    /path/to/your/MVBench/
    ├── json/                              # JSON 标注文件
    │   ├── action_sequence.json
    │   ├── action_prediction.json
    │   ├── action_antonym.json
    │   └── ... (共 20 个 json 文件)
    └── video/                             # 视频文件目录
        ├── star/Charades_v1_480/          # Action Sequence 等任务的视频
        ├── ssv2_video/                    # Action Antonym 的视频
        ├── clevrer/video_validation/      # Object Existence 等任务的视频
        └── ...

  准备完成后，运行评测时设置环境变量：
    VLMEVALKIT_USE_MODELSCOPE=0 swift eval \
      --model your_model \
      --eval_dataset MVBench_8frame \
      --eval_backend VLMEvalKit \
      --eval_limit 10
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile

TYPE_DATA_LIST = {
    'Action Sequence': ('action_sequence.json',
                        'your_data_path/star/Charades_v1_480/', 'video', True),
    'Action Prediction': ('action_prediction.json',
                          'your_data_path/star/Charades_v1_480/', 'video', True),
    'Action Antonym': ('action_antonym.json',
                       'your_data_path/ssv2_video/', 'video', False),
    'Fine-grained Action': ('fine_grained_action.json',
                            'your_data_path/Moments_in_Time_Raw/videos/', 'video', False),
    'Unexpected Action': ('unexpected_action.json',
                          'your_data_path/FunQA_test/test/', 'video', False),
    'Object Existence': ('object_existence.json',
                         'your_data_path/clevrer/video_validation/', 'video', False),
    'Object Interaction': ('object_interaction.json',
                           'your_data_path/star/Charades_v1_480/', 'video', True),
    'Object Shuffle': ('object_shuffle.json',
                       'your_data_path/perception/videos/', 'video', False),
    'Moving Direction': ('moving_direction.json',
                         'your_data_path/clevrer/video_validation/', 'video', False),
    'Action Localization': ('action_localization.json',
                            'your_data_path/sta/sta_video/', 'video', True),
    'Scene Transition': ('scene_transition.json',
                         'your_data_path/scene_qa/video/', 'video', False),
    'Action Count': ('action_count.json',
                     'your_data_path/perception/videos/', 'video', False),
    'Moving Count': ('moving_count.json',
                     'your_data_path/clevrer/video_validation/', 'video', False),
    'Moving Attribute': ('moving_attribute.json',
                         'your_data_path/clevrer/video_validation/', 'video', False),
    'State Change': ('state_change.json',
                     'your_data_path/perception/videos/', 'video', False),
    'Fine-grained Pose': ('fine_grained_pose.json',
                          'your_data_path/nturgbd/', 'video', False),
    'Character Order': ('character_order.json',
                        'your_data_path/perception/videos/', 'video', False),
    'Egocentric Navigation': ('egocentric_navigation.json',
                              'your_data_path/vlnqa/', 'video', False),
    'Episodic Reasoning': ('episodic_reasoning.json',
                           'your_data_path/tvqa/frames_fps3_hq/', 'frame', True),
    'Counterfactual Inference': (
        'counterfactual_inference.json',
        'your_data_path/clevrer/video_validation/', 'video', False),
}

EXPECTED_MD5 = 'fd21d36522cdedd46d84dc46715ad832'


def md5_file(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def step1_unzip_videos(data_dir):
    """解压 video/ 下的 zip 文件"""
    video_dir = os.path.join(data_dir, 'video')
    if not os.path.isdir(video_dir):
        return

    zip_files = [f for f in os.listdir(video_dir) if f.endswith('.zip')]
    if not zip_files:
        print('[Step 1] video/ 目录下没有 zip 文件，跳过解压。')
        return

    for filename in zip_files:
        zip_path = os.path.join(video_dir, filename)
        print(f'[Step 1] 正在解压: {zip_path}')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(video_dir)
    print(f'[Step 1] 解压完成，共处理 {len(zip_files)} 个 zip 文件。')


def step2_move_data0613(data_dir):
    """移动 video/data0613/ 下的文件到正确位置"""
    src_folder = os.path.join(data_dir, 'video', 'data0613')
    if not os.path.exists(src_folder):
        print('[Step 2] video/data0613/ 不存在，跳过文件移动。')
        return

    moved = 0
    for subdir in os.listdir(src_folder):
        subdir_path = os.path.join(src_folder, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for subsubdir in os.listdir(subdir_path):
            subsubdir_path = os.path.join(subdir_path, subsubdir)
            if not os.path.isdir(subsubdir_path):
                continue
            for item in os.listdir(subsubdir_path):
                item_path = os.path.join(subsubdir_path, item)
                target_folder = os.path.join(data_dir, 'video', subdir, subsubdir)
                os.makedirs(target_folder, exist_ok=True)
                target_path = os.path.join(target_folder, item)
                try:
                    shutil.move(item_path, target_path)
                    moved += 1
                except Exception as e:
                    print(f'  警告: 移动失败 {item_path} -> {target_path}: {e}')
    print(f'[Step 2] 文件移动完成，共移动 {moved} 个文件。')


def step3_generate_tsv(data_dir, skip_missing=False):
    """从 json/ 目录生成 MVBench.tsv（与 VLMEvalKit 逻辑一致）"""
    import pandas as pd

    json_data_dir = os.path.join(data_dir, 'json')
    if not os.path.isdir(json_data_dir):
        print(f'[Step 3] 错误: json/ 目录不存在: {json_data_dir}')
        sys.exit(1)

    data_list = []
    missing_videos = []

    for task_type, (json_file, prefix_template, data_type, bound) in TYPE_DATA_LIST.items():
        json_path = os.path.join(json_data_dir, json_file)
        if not os.path.exists(json_path):
            print(f'  警告: JSON 文件不存在: {json_path}，跳过任务 {task_type}')
            continue

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        prefix = prefix_template.replace('your_data_path', 'video')
        for data in json_data:
            video_path = os.path.join(data_dir, prefix, data['video'])
            if os.path.exists(video_path):
                data_list.append({
                    'task_type': task_type,
                    'prefix': prefix,
                    'data_type': data_type,
                    'bound': bound,
                    'start': data.get('start'),
                    'end': data.get('end'),
                    'video': data['video'],
                    'question': data['question'],
                    'answer': data['answer'],
                    'candidates': data['candidates']
                })
            else:
                missing_videos.append((task_type, video_path))
                if not skip_missing:
                    print(f'\n  错误: 视频文件不存在: {video_path}')
                    print(f'  任务类型: {task_type}')
                    print('\n  提示: 如果你想跳过缺失的视频，请使用 --skip-missing 参数。')
                    print('  注意: 跳过缺失视频会导致 TSV 的 MD5 与官方不一致，')
                    print('  需要额外的处理步骤（脚本会自动处理）。')
                    sys.exit(1)

    if missing_videos and skip_missing:
        print(f'  警告: 共有 {len(missing_videos)} 个视频文件缺失，已跳过。')
        task_counts = {}
        for task_type, _ in missing_videos:
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        for task_type, count in task_counts.items():
            print(f'    {task_type}: 缺失 {count} 个视频')

    data_df = pd.DataFrame(data_list)
    data_df = data_df.assign(index=range(len(data_df)))

    tsv_path = os.path.join(data_dir, 'MVBench.tsv')
    data_df.to_csv(tsv_path, sep='\t', index=False)

    file_md5 = md5_file(tsv_path)
    print(f'[Step 3] TSV 生成完成: {tsv_path}')
    print(f'         共 {len(data_list)} 条数据')
    print(f'         MD5: {file_md5}')

    if file_md5 == EXPECTED_MD5:
        print('         ✓ MD5 与官方一致，数据完整！')
        return True
    else:
        print(f'         ✗ MD5 与官方不一致（期望: {EXPECTED_MD5}）')
        print('         这通常是因为缺少部分视频文件。')
        return False


def step4_setup_hf_cache(data_dir):
    """在 HuggingFace 缓存目录下创建指向本地数据的缓存结构"""
    cache_root = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
    for env_var in ['HUGGINGFACE_HUB_CACHE', 'HF_HOME']:
        if env_var in os.environ and os.path.exists(os.environ[env_var]):
            val = os.environ[env_var]
            if val.split('/')[-1] == 'hub':
                cache_root = val
            else:
                cache_root = os.path.join(val, 'hub')
            break

    repo_cache_dir = os.path.join(cache_root, 'datasets--OpenGVLab--MVBench')
    refs_dir = os.path.join(repo_cache_dir, 'refs')
    snapshots_dir = os.path.join(repo_cache_dir, 'snapshots')
    fake_hash = 'offline_local_data'
    snapshot_path = os.path.join(snapshots_dir, fake_hash)

    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)

    refs_main = os.path.join(refs_dir, 'main')
    with open(refs_main, 'w') as f:
        f.write(fake_hash)

    if os.path.islink(snapshot_path):
        os.unlink(snapshot_path)
    elif os.path.isdir(snapshot_path):
        shutil.rmtree(snapshot_path)

    data_dir_abs = os.path.abspath(data_dir)
    os.symlink(data_dir_abs, snapshot_path)

    print('[Step 4] HuggingFace 缓存结构已创建:')
    print(f'         缓存目录: {repo_cache_dir}')
    print(f'         refs/main -> {fake_hash}')
    print(f'         snapshots/{fake_hash} -> {data_dir_abs}')


def step5_patch_integrity_check(data_dir):
    """
    当 MD5 不匹配时，生成包装脚本跳过完整性校验。
    """
    data_dir_abs = os.path.abspath(data_dir)

    print('\n[额外步骤] MD5 不一致的处理方案:')
    print('  由于你的数据集可能缺少部分视频（如 NTURGB-D 已被官方移除），')
    print('  VLMEvalKit 的完整性校验会失败。')
    print()
    print('  推荐方案: 使用本脚本生成的包装脚本 run_eval_offline.py 运行评测。')
    print()

    wrapper_path = os.path.join(data_dir, 'run_eval_offline.py')
    wrapper_content = _build_wrapper_script(data_dir_abs, wrapper_path)
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)
    print('  已生成包装脚本: %s' % wrapper_path)
    print('  该脚本会自动 monkey-patch MVBench 的数据加载，跳过完整性校验和网络下载。')


def _build_wrapper_script(data_dir_abs, wrapper_path):
    """构建包装脚本内容"""
    return '''#!/usr/bin/env python3
"""
离线评测 MVBench 的包装脚本。
自动跳过 VLMEvalKit 的完整性校验，直接使用本地数据。

使用方法（根据你的实际参数修改）：
  python %(wrapper_path)s \\
    --model Qwen/Qwen2.5-VL-3B-Instruct \\
    --eval_dataset MVBench_8frame \\
    --eval_backend VLMEvalKit \\
    --eval_limit 10

所有参数与 swift eval 完全一致。
"""
import os
import sys

LOCAL_DATA_PATH = '%(data_dir)s'

os.environ['VLMEVALKIT_USE_MODELSCOPE'] = '0'

import vlmeval.dataset.mvbench as _mvbench_module


class _PatchedMVBench(_mvbench_module.MVBench):
    def prepare_dataset(self, dataset_name='MVBench', repo_id='OpenGVLab/MVBench'):
        import torchvision.transforms as T

        pth = LOCAL_DATA_PATH
        data_file = os.path.join(pth, f'{dataset_name}.tsv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f'TSV file not found: {data_file}\\n'
                'Please run prepare_mvbench_offline.py first.'
            )

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        self.nframe = 8
        self.frame_fps = 3

        from vlmeval.dataset.mvbench import Stack, ToTorchFormatTensor
        self.transform = T.Compose([Stack(), ToTorchFormatTensor()])

        return dict(root=pth, data_file=data_file)


_mvbench_module.MVBench = _PatchedMVBench

from vlmeval.dataset.video_dataset_config import mvbench_dataset
from functools import partial
mvbench_dataset['MVBench_8frame'] = partial(_PatchedMVBench, dataset='MVBench', nframe=8)
mvbench_dataset['MVBench_64frame'] = partial(_PatchedMVBench, dataset='MVBench', nframe=64)

from vlmeval.dataset import supported_video_datasets
supported_video_datasets['MVBench_8frame'] = mvbench_dataset['MVBench_8frame']
supported_video_datasets['MVBench_64frame'] = mvbench_dataset['MVBench_64frame']

from swift.pipelines import eval_main
eval_main()
''' % {'data_dir': data_dir_abs, 'wrapper_path': wrapper_path}


def main():
    parser = argparse.ArgumentParser(
        description='离线环境下准备 MVBench 数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 基本用法
  python prepare_mvbench_offline.py --data-dir /data/MVBench

  # 跳过缺失的视频文件
  python prepare_mvbench_offline.py --data-dir /data/MVBench --skip-missing

准备完成后运行评测:
  # 如果 MD5 匹配（数据完整）:
  VLMEVALKIT_USE_MODELSCOPE=0 swift eval \\
    --model Qwen/Qwen2.5-VL-3B-Instruct \\
    --eval_dataset MVBench_8frame \\
    --eval_backend VLMEvalKit \\
    --eval_limit 10

  # 如果 MD5 不匹配（数据不完整），使用生成的包装脚本:
  python /data/MVBench/run_eval_offline.py \\
    --model Qwen/Qwen2.5-VL-3B-Instruct \\
    --eval_dataset MVBench_8frame \\
    --eval_backend VLMEvalKit \\
    --eval_limit 10
        ''')
    parser.add_argument('--data-dir', required=True,
                        help='本地 MVBench 数据集目录（包含 json/ 和 video/ 子目录）')
    parser.add_argument('--skip-missing', action='store_true',
                        help='跳过缺失的视频文件（TSV 的 MD5 将与官方不一致）')

    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    print('═══════════════════════════════════════════════════')
    print('  MVBench 离线数据准备工具')
    print(f'  数据目录: {data_dir}')
    print('═══════════════════════════════════════════════════')
    print()

    if not os.path.isdir(data_dir):
        print(f'错误: 目录不存在: {data_dir}')
        sys.exit(1)

    json_dir = os.path.join(data_dir, 'json')
    video_dir = os.path.join(data_dir, 'video')

    if not os.path.isdir(json_dir):
        print(f'错误: json/ 子目录不存在: {json_dir}')
        sys.exit(1)
    if not os.path.isdir(video_dir):
        print(f'错误: video/ 子目录不存在: {video_dir}')
        sys.exit(1)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print(f'找到 {len(json_files)} 个 JSON 文件。')
    print()

    # Step 1: 解压 zip 文件
    step1_unzip_videos(data_dir)
    print()

    # Step 2: 移动 data0613 文件
    step2_move_data0613(data_dir)
    print()

    # Step 3: 生成 TSV
    md5_matched = step3_generate_tsv(data_dir, skip_missing=args.skip_missing)
    print()

    # Step 4: 设置 HF 缓存
    step4_setup_hf_cache(data_dir)
    print()

    # 根据 MD5 是否匹配给出不同的后续指引
    if md5_matched:
        print('═══════════════════════════════════════════════════')
        print('  ✓ 准备完成！数据完整，可以直接运行评测。')
        print('═══════════════════════════════════════════════════')
        print()
        print('运行评测命令（根据你的实际参数修改）:')
        print()
        print('  VLMEVALKIT_USE_MODELSCOPE=0 swift eval \\')
        print('    --model Qwen/Qwen2.5-VL-3B-Instruct \\')
        print('    --eval_dataset MVBench_8frame \\')
        print('    --eval_backend VLMEvalKit \\')
        print('    --eval_limit 10')
    else:
        # 生成包装脚本
        step5_patch_integrity_check(data_dir)
        print()
        print('═══════════════════════════════════════════════════')
        print('  ⚠ 准备完成，但数据不完整（部分视频缺失）。')
        print('  请使用生成的包装脚本运行评测。')
        print('═══════════════════════════════════════════════════')
        print()
        wrapper_path = os.path.join(data_dir, 'run_eval_offline.py')
        print('运行评测命令:')
        print()
        print(f'  python {wrapper_path} \\')
        print('    --model Qwen/Qwen2.5-VL-3B-Instruct \\')
        print('    --eval_dataset MVBench_8frame \\')
        print('    --eval_backend VLMEvalKit \\')
        print('    --eval_limit 10')

    print()


if __name__ == '__main__':
    main()
