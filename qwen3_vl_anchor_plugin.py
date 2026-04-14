"""Custom qwen3-vl template with anchors + anchor_type.

Usage:
    swift sft \
      --model Qwen/Qwen3-VL-4B-Instruct \
      --dataset /path/to/train.jsonl \
      --external_plugins /workspace/qwen3_vl_anchor_plugin.py \
      --template qwen3_vl_anchor \
      --remove_unused_columns false

anchor_type:
    0 -> no processing
    1 -> crop by anchors
    2 -> draw rectangle by anchors
"""

import json
import os
import tempfile
from hashlib import md5
from typing import Any, Optional, Tuple

from PIL import Image, ImageDraw

from swift.template import register_template
from swift.template.base import Template
from swift.template.templates.qwen import Qwen3VLTemplate, QwenTemplateMeta
from swift.utils import get_logger

logger = get_logger()
TEMPLATE_TYPE = 'qwen3_vl_anchor'


def _json_loads_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return value


def _pick_index(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(index, value.get(str(index)))
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if isinstance(value[0], (list, tuple, dict)):
            return value[index] if index < len(value) else None
        return value
    return value


def _pick_media_value(raw: Any, media_type: str, index: int) -> Any:
    raw = _json_loads_maybe(raw)
    if isinstance(raw, dict):
        media_key = 'images' if media_type == 'image' else 'videos'
        if media_key in raw:
            return _pick_index(raw[media_key], index)
        if media_type in raw:
            return _pick_index(raw[media_type], index)
    return _pick_index(raw, index)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_anchor(anchor: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if anchor is None:
        return None
    anchor = _json_loads_maybe(anchor)

    if isinstance(anchor, dict):
        if 'bbox' in anchor:
            anchor = anchor['bbox']
        else:
            anchor = [anchor.get('x1'), anchor.get('y1'), anchor.get('x2'), anchor.get('y2')]

    if not isinstance(anchor, (list, tuple)) or len(anchor) < 4:
        return None

    x1, y1, x2, y2 = anchor[:4]
    try:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    except Exception:
        return None

    # If coords are in [0,1], treat as normalized.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _apply_anchor_to_image_obj(image: Image.Image, anchor: Any, anchor_type: int) -> Image.Image:
    box = _normalize_anchor(anchor, image.width, image.height)
    if box is None:
        return image
    if anchor_type == 1:
        return image.crop(box)
    if anchor_type == 2:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        line_width = max(2, min(image.width, image.height) // 200)
        draw.rectangle(box, outline='red', width=line_width)
    return image


def _load_image_flexible(image: Any) -> Image.Image:
    return Template._load_image(image, load_images=True)


def _apply_anchor_to_video_path(video_path: str, anchor: Any, anchor_type: int) -> Any:
    """Best-effort local video processing; fallback to source path on errors."""
    if not isinstance(video_path, str) or not os.path.isfile(video_path):
        return video_path
    try:
        import imageio.v3 as iio
    except Exception:
        logger.warning_once('imageio is unavailable, skip video anchor processing.')
        return video_path
    try:
        frames = iio.imread(video_path)
        if getattr(frames, 'ndim', 0) != 4:
            return video_path
    except Exception as e:
        logger.warning_once(f'Failed reading video for anchors, fallback. err={e}')
        return video_path

    h, w = frames.shape[1], frames.shape[2]
    box = _normalize_anchor(anchor, w, h)
    if box is None:
        return video_path
    x1, y1, x2, y2 = box

    if anchor_type == 1:
        frames = frames[:, y1:y2, x1:x2, :]
    elif anchor_type == 2:
        out_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            line_width = max(2, min(img.width, img.height) // 200)
            draw.rectangle(box, outline='red', width=line_width)
            out_frames.append(img)
        frames = out_frames
    else:
        return video_path

    try:
        meta = iio.immeta(video_path)
        fps = int(meta.get('fps', 25)) if isinstance(meta, dict) else 25
    except Exception:
        fps = 25

    digest = md5(
        f'{video_path}|{anchor_type}|{json.dumps(anchor, ensure_ascii=False, sort_keys=True)}'.encode('utf-8')
    ).hexdigest()[:10]
    out_dir = os.path.join(tempfile.gettempdir(), 'swift_anchor_media')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{os.path.basename(video_path)}.{digest}.mp4')
    if not os.path.exists(out_path):
        try:
            iio.imwrite(out_path, frames, fps=fps)
        except Exception as e:
            logger.warning_once(f'Failed writing processed video, fallback. err={e}')
            return video_path
    return out_path


def _apply_anchor_to_video(video: Any, anchor: Any, anchor_type: int) -> Any:
    if anchor_type == 0:
        return video
    if isinstance(video, (list, tuple)) and video:
        # Video represented by frames.
        processed = []
        for frame in video:
            try:
                img = _load_image_flexible(frame)
                img = _apply_anchor_to_image_obj(img, anchor, anchor_type)
                processed.append(img)
            except Exception:
                processed.append(frame)
        return processed
    return _apply_anchor_to_video_path(video, anchor, anchor_type)


class Qwen3VLAnchorTemplate(Qwen3VLTemplate):

    def replace_tag(self, media_type, index, inputs):
        if media_type in {'image', 'video'}:
            anchor_type_raw = _pick_media_value(inputs.extra_kwargs.get('anchor_type', 0), media_type, index)
            anchor_type = _to_int(anchor_type_raw, default=0)
            if anchor_type not in {0, 1, 2}:
                logger.warning_once(f'Invalid anchor_type={anchor_type}, fallback to 0')
                anchor_type = 0

            anchor = _pick_media_value(inputs.extra_kwargs.get('anchors'), media_type, index)
            if anchor is not None and anchor_type in {1, 2}:
                if media_type == 'image':
                    image = _load_image_flexible(inputs.images[index])
                    inputs.images[index] = _apply_anchor_to_image_obj(image, anchor, anchor_type)
                else:
                    inputs.videos[index] = _apply_anchor_to_video(inputs.videos[index], anchor, anchor_type)

            # Optional pass-through for custom downstream processors.
            if anchor is not None:
                inputs.mm_processor_kwargs.setdefault(f'{media_type}_anchors', []).append(anchor)
                inputs.mm_processor_kwargs.setdefault(f'{media_type}_anchor_type', anchor_type)

        return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs):
        # Fallback if processor does not accept custom anchor kwargs.
        try:
            return super()._encode(inputs)
        except TypeError as e:
            keys = [k for k in list(inputs.mm_processor_kwargs.keys()) if 'anchor' in k]
            if keys and ('unexpected keyword' in str(e) or 'got an unexpected keyword argument' in str(e)):
                logger.warning('Processor does not support custom anchor kwargs %s; retry without them.', keys)
                for k in keys:
                    inputs.mm_processor_kwargs.pop(k, None)
                return super()._encode(inputs)
            raise


register_template(
    QwenTemplateMeta(
        TEMPLATE_TYPE,
        template_cls=Qwen3VLAnchorTemplate,
        default_system=None,
        thinking_prefix='<think>\n',
    ),
    exist_ok=True,
)

