#!/usr/bin/env python3
"""LoRA inference script for qwen3_vl_anchor_plugin.

Supports:
- Standard `messages` JSON/JSONL
- query-response JSON/JSONL (with optional `history` / `system`), converted via
  the same rules as ms-swift ResponsePreprocessor
- External anchor fields (`anchors`, `anchor_type`, `shape`, etc.) forwarded to
  the template through `extra_kwargs`

Example:
    python qwen3_vl_anchor_infer.py \
      --adapter-path /path/to/lora_ckpt \
      --input-file /path/to/infer.jsonl \
      --output-file /path/to/infer_output.jsonl
"""

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

from swift import BaseArguments, RequestConfig, TransformersEngine, get_template
from swift.template import history_to_messages

TRAINING_ONLY_KEYS = {
    'query',
    'prompt',
    'text',
    'input',
    'instruction',
    'question',
    'problem',
    'response',
    'answer',
    'output',
    'targets',
    'target',
    'answer_key',
    'answers',
    'solution',
    'completion',
    'content',
    'history',
    'from',
    'label',
    'labels',
    'label_type',
    'cls_label',
    'system',
    'system_prompt',
}

QUERY_KEYS = ('query', 'prompt', 'input', 'instruction', 'question', 'problem', 'text')
RESPONSE_KEYS = ('response', 'answer', 'output', 'targets', 'target', 'answer_key', 'answers', 'solution', 'text',
                 'completion', 'content')
SYSTEM_KEYS = ('system', 'system_prompt')


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Infer JSON/JSONL with Qwen3-VL LoRA + anchor plugin.')
    parser.add_argument('--adapter-path', type=str, required=True, help='LoRA checkpoint directory.')
    parser.add_argument('--input-file', type=str, required=True, help='Input .json or .jsonl file.')
    parser.add_argument('--output-file', type=str, required=True, help='Output .json or .jsonl file.')
    parser.add_argument(
        '--external-plugins',
        '--external_plugins',
        '--external-plugin',
        dest='external_plugins',
        nargs='+',
        default=['qwen3_vl_anchor_plugin.py'],
        help='One or multiple external plugin paths.')
    parser.add_argument(
        '--remove-unused-columns',
        '--remove_unused_columns',
        type=str2bool,
        default=True,
        help='Forwarded to get_template(remove_unused_columns=...). Recommend true for inference.')
    parser.add_argument('--model', type=str, default=None, help='Base model path/id. Defaults to adapter args.json.')
    parser.add_argument('--template', type=str, default=None, help='Template name. Defaults to adapter args.json.')
    parser.add_argument('--system', type=str, default=None, help='Override system prompt.')
    parser.add_argument('--batch-size', type=int, default=4, help='Inference batch size.')
    parser.add_argument('--max-new-tokens', type=int, default=512, help='Max generated tokens.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature.')
    parser.add_argument('--top-p', type=float, default=1.0, help='Top-p sampling.')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling.')
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty.')
    parser.add_argument('--seed', type=int, default=42, help='Generation seed.')
    return parser.parse_args()


def import_external_plugin(plugin_path: Path) -> None:
    plugin_path = plugin_path.resolve()
    if not plugin_path.exists():
        raise FileNotFoundError(f'Plugin file not found: {plugin_path}')
    module_name = f'qwen3_vl_anchor_plugin_{abs(hash(str(plugin_path)))}'
    spec = importlib.util.spec_from_file_location(module_name, str(plugin_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load plugin: {plugin_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


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
                    raise ValueError(f'Each JSONL row must be an object, got {type(row)} at line {line_no}')
                samples.append(row)
        return samples

    if suffix != '.json':
        raise ValueError(f'Unsupported input extension: {input_path.suffix}. Use .json or .jsonl')

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
        raise ValueError(f'Invalid JSON top-level type: {type(data)}')

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f'Each sample must be an object, got {type(sample)} at index {idx}')
    return samples


def _pop_first(item: Dict[str, Any], keys: tuple) -> Any:
    for key in keys:
        if key in item:
            return item.pop(key)
    return None


def _query_response_to_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert query-response (+ optional history/system) to messages for inference."""
    history = _pop_first(item, ('history', )) or []
    if isinstance(history, str):
        history = ast.literal_eval(history)
    if not isinstance(history, list):
        raise ValueError(f'`history` must be a list, got {type(history)}')

    query = _pop_first(item, QUERY_KEYS)
    _pop_first(item, RESPONSE_KEYS)  # training label; do not feed into generation
    system = _pop_first(item, SYSTEM_KEYS)

    if query is None and not history:
        raise ValueError('Sample missing `messages` and no usable `query`/`history` fields.')

    if query is not None:
        history = list(history)
        history.append([query, None])

    return history_to_messages(history, system)


def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    item = copy.deepcopy(sample)
    if 'anchors' not in item and 'anchor' in item:
        item['anchors'] = item.pop('anchor')
    for singular, plural in (('image', 'images'), ('video', 'videos'), ('audio', 'audios')):
        if plural not in item and singular in item:
            value = item.pop(singular)
            item[plural] = value if isinstance(value, list) else [value]

    if 'messages' not in item:
        item['messages'] = _query_response_to_messages(item)
    else:
        for key in QUERY_KEYS + RESPONSE_KEYS + SYSTEM_KEYS + ('history', ):
            item.pop(key, None)

    for key in TRAINING_ONLY_KEYS:
        item.pop(key, None)

    if not isinstance(item['messages'], list) or not item['messages']:
        raise ValueError(f'`messages` must be a non-empty list. Got: {item.get("messages")}')
    return item


def save_outputs(output_path: Path, rows: List[Dict[str, Any]]) -> None:
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


def build_engine(args: argparse.Namespace) -> TransformersEngine:
    adapter_path = Path(args.adapter_path).resolve()
    train_args = BaseArguments.from_pretrained(str(adapter_path))
    model = args.model or getattr(train_args, 'model', None)
    if not model:
        raise ValueError('Cannot determine base model. Please provide --model.')
    template_type = args.template or getattr(train_args, 'template', None) or 'qwen3_vl_anchor'
    default_system = args.system if args.system is not None else getattr(train_args, 'system', None)

    engine = TransformersEngine(model, adapters=[str(adapter_path)], max_batch_size=max(1, args.batch_size))
    engine.template = get_template(
        engine.processor,
        default_system=default_system,
        template_type=template_type,
        remove_unused_columns=args.remove_unused_columns)
    return engine


def main() -> None:
    args = parse_args()
    for plugin in args.external_plugins:
        import_external_plugin(Path(plugin))
    engine = build_engine(args)

    samples = load_samples(Path(args.input_file))
    infer_requests = [normalize_sample(sample) for sample in samples]
    request_config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed)

    outputs: List[Dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(infer_requests), batch_size):
        batch = infer_requests[start:start + batch_size]
        responses = engine.infer(batch, request_config, use_tqdm=False)
        for request_item, resp in zip(batch, responses):
            output_item = copy.deepcopy(request_item)
            output_item['response'] = resp.choices[0].message.content
            output_item['finish_reason'] = resp.choices[0].finish_reason
            if resp.usage is not None:
                output_item['usage'] = {
                    'prompt_tokens': resp.usage.prompt_tokens,
                    'completion_tokens': resp.usage.completion_tokens,
                    'total_tokens': resp.usage.total_tokens,
                }
            outputs.append(output_item)

    save_outputs(Path(args.output_file), outputs)
    print(f'Inference finished. Saved {len(outputs)} rows to: {args.output_file}')


if __name__ == '__main__':
    main()
