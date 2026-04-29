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
from typing import Any, Optional, Tuple

import numpy as np
import torch
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


def _parse_shape_wh(shape: Any) -> Optional[Tuple[int, int]]:
    shape = _json_loads_maybe(shape)
    if shape is None:
        return None
    if isinstance(shape, dict):
        if 'shape' in shape:
            shape = shape['shape']
        elif 'size' in shape:
            shape = shape['size']
        elif 'w' in shape and 'h' in shape:
            shape = [shape['w'], shape['h']]
        elif 'width' in shape and 'height' in shape:
            shape = [shape['width'], shape['height']]
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    try:
        w = int(shape[0])
        h = int(shape[1])
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return w, h


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


def _rescale_box(box: Tuple[int, int, int, int], src_w: int, src_h: int, dst_w: int, dst_h: int) -> Tuple[int, int, int, int]:
    if src_w <= 0 or src_h <= 0:
        return box
    x1, y1, x2, y2 = box
    sx = dst_w / src_w
    sy = dst_h / src_h
    nx1 = int(round(x1 * sx))
    ny1 = int(round(y1 * sy))
    nx2 = int(round(x2 * sx))
    ny2 = int(round(y2 * sy))
    nx1 = max(0, min(nx1, dst_w - 1))
    ny1 = max(0, min(ny1, dst_h - 1))
    nx2 = max(1, min(nx2, dst_w))
    ny2 = max(1, min(ny2, dst_h))
    return nx1, ny1, nx2, ny2


def _apply_anchor_to_image_obj(image: Image.Image,
                               anchor: Any,
                               anchor_type: int,
                               *,
                               shape_wh: Optional[Tuple[int, int]] = None,
                               resize_after_crop: bool = False,
                               resize_target: Optional[Tuple[int, int]] = None) -> Image.Image:
    curr_w, curr_h = image.width, image.height
    anchor_ref_w, anchor_ref_h = shape_wh or (curr_w, curr_h)
    box = _normalize_anchor(anchor, anchor_ref_w, anchor_ref_h)
    if box is None:
        return image
    if (anchor_ref_w, anchor_ref_h) != (curr_w, curr_h):
        box = _rescale_box(box, anchor_ref_w, anchor_ref_h, curr_w, curr_h)
    if anchor_type == 1:
        image = image.crop(box)
        if resize_after_crop:
            target_w, target_h = resize_target or (curr_w, curr_h)
            image = image.resize((target_w, target_h), Image.BICUBIC)
        return image
    if anchor_type == 2:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        line_width = max(2, min(image.width, image.height) // 200)
        draw.rectangle(box, outline='red', width=line_width)
    return image


def _load_image_flexible(image: Any) -> Image.Image:
    return Template._load_image(image, load_images=True)


def _is_channel_last_video(arr: Any) -> bool:
    return arr.ndim == 4 and arr.shape[-1] in (1, 3, 4)


def _is_channel_first_video(arr: Any) -> bool:
    return arr.ndim == 4 and arr.shape[1] in (1, 3, 4)


def _apply_anchor_to_numpy_video(video: np.ndarray,
                                 anchor: Any,
                                 anchor_type: int,
                                 *,
                                 shape_wh: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if video.ndim != 4:
        return video
    channel_last = _is_channel_last_video(video)
    if not channel_last and not _is_channel_first_video(video):
        return video

    if channel_last:
        curr_h, curr_w = video.shape[1], video.shape[2]
    else:
        curr_h, curr_w = video.shape[2], video.shape[3]
    anchor_ref_w, anchor_ref_h = shape_wh or (curr_w, curr_h)
    box = _normalize_anchor(anchor, anchor_ref_w, anchor_ref_h)
    if box is None:
        return video
    if (anchor_ref_w, anchor_ref_h) != (curr_w, curr_h):
        box = _rescale_box(box, anchor_ref_w, anchor_ref_h, curr_w, curr_h)
    x1, y1, x2, y2 = box

    if anchor_type not in (1, 2):
        return video

    out = []
    for frame in video:
        if channel_last:
            frame_hwc = frame
        else:
            frame_hwc = np.transpose(frame, (1, 2, 0))
        frame_hwc = np.asarray(frame_hwc, dtype=np.uint8)
        img = Image.fromarray(frame_hwc)
        img = _apply_anchor_to_image_obj(
            img,
            anchor,
            anchor_type,
            shape_wh=(curr_w, curr_h),
            resize_after_crop=(anchor_type == 1),
            resize_target=(curr_w, curr_h))
        frame_out = np.asarray(img)
        if not channel_last:
            frame_out = np.transpose(frame_out, (2, 0, 1))
        out.append(frame_out)
    return np.ascontiguousarray(np.stack(out, axis=0))


def _apply_anchor_to_torch_video(video: torch.Tensor,
                                 anchor: Any,
                                 anchor_type: int,
                                 *,
                                 shape_wh: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    if video.ndim != 4:
        return video
    if anchor_type not in (1, 2):
        return video
    video_np = video.detach().cpu().numpy()
    out_np = _apply_anchor_to_numpy_video(video_np, anchor, anchor_type, shape_wh=shape_wh)
    out = torch.from_numpy(out_np).to(video.device)
    if out.dtype != video.dtype:
        out = out.to(video.dtype)
    return out


def _apply_anchor_to_video(video: Any,
                           anchor: Any,
                           anchor_type: int,
                           *,
                           shape_wh: Optional[Tuple[int, int]] = None) -> Any:
    if anchor_type == 0:
        return video
    if isinstance(video, torch.Tensor):
        return _apply_anchor_to_torch_video(video, anchor, anchor_type, shape_wh=shape_wh)
    if isinstance(video, np.ndarray):
        return _apply_anchor_to_numpy_video(video, anchor, anchor_type, shape_wh=shape_wh)
    if isinstance(video, (list, tuple)) and video:
        # Video represented by frame list.
        processed = []
        for frame in video:
            try:
                img = _load_image_flexible(frame)
                img = _apply_anchor_to_image_obj(
                    img,
                    anchor,
                    anchor_type,
                    shape_wh=shape_wh,
                    resize_after_crop=(anchor_type == 1),
                    resize_target=shape_wh or (img.width, img.height))
                processed.append(np.asarray(img))
            except Exception:
                processed.append(frame)
        return processed
    return video


def _sync_video_metadata(inputs, index: int) -> None:
    """Remove stale video metadata after in-memory frame edits.

    Qwen3-VL (v3) stores `video_metadata` during `replace_tag(fetch_video)`.
    Once we crop/resize frames in memory, old metadata can become inconsistent.
    We clear it and let the later processor call in `_encode` consume updated frames.
    """
    mm_kwargs = inputs.mm_processor_kwargs
    video_metadata = mm_kwargs.get('video_metadata', None)
    if video_metadata is not None:
        # Avoid mixing stale/new entries across multi-video samples after frame edits.
        mm_kwargs.pop('video_metadata', None)


class Qwen3VLAnchorTemplate(Qwen3VLTemplate):

    def replace_tag(self, media_type, index, inputs):
        anchor = None
        anchor_type = 0
        if media_type in {'image', 'video'}:
            anchor_type_raw = _pick_media_value(inputs.extra_kwargs.get('anchor_type', 0), media_type, index)
            anchor_type = _to_int(anchor_type_raw, default=0)
            if anchor_type not in {0, 1, 2}:
                logger.warning_once(f'Invalid anchor_type={anchor_type}, fallback to 0')
                anchor_type = 0

            anchor = _pick_media_value(inputs.extra_kwargs.get('anchors'), media_type, index)
            shape_raw = _pick_media_value(inputs.extra_kwargs.get('shape'), media_type, index)
            shape_wh = _parse_shape_wh(shape_raw)
            if anchor is not None and anchor_type in {1, 2}:
                if media_type == 'image':
                    image = _load_image_flexible(inputs.images[index])
                    inputs.images[index] = _apply_anchor_to_image_obj(
                        image,
                        anchor,
                        anchor_type,
                        shape_wh=shape_wh,
                        resize_after_crop=(anchor_type == 1),
                        resize_target=shape_wh)

            # Optional pass-through for custom downstream processors.
            if anchor is not None:
                inputs.mm_processor_kwargs.setdefault(f'{media_type}_anchors', []).append(anchor)
                inputs.mm_processor_kwargs.setdefault(f'{media_type}_anchor_type', anchor_type)

        contexts = super().replace_tag(media_type, index, inputs)
        if media_type == 'video' and anchor is not None and anchor_type in {1, 2}:
            shape_raw = _pick_media_value(inputs.extra_kwargs.get('shape'), media_type, index)
            shape_wh = _parse_shape_wh(shape_raw)
            # Apply anchor ops on extracted in-memory frames/tensors.
            inputs.videos[index] = _apply_anchor_to_video(
                inputs.videos[index], anchor, anchor_type, shape_wh=shape_wh)
            # Metadata from pre-edit frames may be stale; clear for downstream re-processing.
            _sync_video_metadata(inputs, index)
        return contexts

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

