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
import math
import os
import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from swift.template import register_template
from swift.template.base import Template
from swift.template.templates.qwen import Qwen3VLTemplate, QwenTemplateMeta
from swift.utils import get_logger

logger = get_logger()
TEMPLATE_TYPE = 'qwen3_vl_anchor'
ANCHOR_TYPE_NONE = 0
ANCHOR_TYPE_CROP = 1
ANCHOR_TYPE_DRAW = 2
VALID_ANCHOR_TYPES = {ANCHOR_TYPE_NONE, ANCHOR_TYPE_CROP, ANCHOR_TYPE_DRAW}


def _has_anchor_operation(anchor: Any, anchor_type: int) -> bool:
    return anchor is not None and anchor_type in {ANCHOR_TYPE_CROP, ANCHOR_TYPE_DRAW}


def _normalize_anchor_type(value: Any) -> int:
    anchor_type = _to_int(value, default=ANCHOR_TYPE_NONE)
    if anchor_type not in VALID_ANCHOR_TYPES:
        logger.warning_once(f'Invalid anchor_type={anchor_type}, fallback to {ANCHOR_TYPE_NONE}')
        return ANCHOR_TYPE_NONE
    return anchor_type


def _to_bool(value: Any, default: bool = False) -> bool:
    value = _json_loads_maybe(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if text in {'0', 'false', 'no', 'n', 'off', ''}:
            return False
    return bool(value)


def _is_noop_anchor(anchor: Any) -> bool:
    if not isinstance(anchor, (list, tuple)) or len(anchor) < 4:
        return False
    try:
        a, b, c, d = (float(anchor[0]), float(anchor[1]), float(anchor[2]), float(anchor[3]))
    except Exception:
        return False
    return abs(a) < 1e-8 and abs(b) < 1e-8 and abs(c) < 1e-8 and abs(d) < 1e-8


def _anchor_type_name(anchor_type: int) -> str:
    if anchor_type == ANCHOR_TYPE_CROP:
        return 'crop'
    if anchor_type == ANCHOR_TYPE_DRAW:
        return 'draw'
    return 'none'


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


def _normalize_anchor(anchor: Any,
                      width: int,
                      height: int,
                      *,
                      anchor_format: str = 'auto') -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(anchor, (list, tuple)) or len(anchor) < 4:
        return None

    _ = anchor_format  # keep signature compatibility
    x1, y1, x2, y2 = anchor[:4]
    try:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    except Exception:
        return None

    # Absolute xyxy coordinates only.
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


def _to_draw_box(box: Tuple[int, int, int, int], width: int, height: int, line_width: int) -> Tuple[int, int, int, int]:
    """Convert crop-style box [x1,y1,x2,y2) to visible draw box inside image."""
    x1, y1, x2, y2 = box
    # PIL draw.rectangle uses inclusive max corner. Convert first.
    left = max(0, min(x1, width - 1))
    top = max(0, min(y1, height - 1))
    right = max(left, min(x2 - 1, width - 1))
    bottom = max(top, min(y2 - 1, height - 1))

    # Keep border-visible: if touching boundary, move inward by half line width.
    inset = max(1, line_width // 2)
    if left == 0:
        left = min(inset, width - 1)
    if top == 0:
        top = min(inset, height - 1)
    if right == width - 1:
        right = max(left, width - 1 - inset)
    if bottom == height - 1:
        bottom = max(top, height - 1 - inset)
    return left, top, right, bottom


def _apply_anchor_to_image_obj(image: Image.Image,
                               anchor: Any,
                               anchor_type: int,
                               *,
                               shape_wh: Optional[Tuple[int, int]] = None,
                               resize_after_crop: bool = False,
                               resize_target: Optional[Tuple[int, int]] = None,
                               anchor_format: str = 'auto') -> Image.Image:
    curr_w, curr_h = image.width, image.height
    anchor_ref_w, anchor_ref_h = shape_wh or (curr_w, curr_h)
    box = _normalize_anchor(anchor, anchor_ref_w, anchor_ref_h, anchor_format=anchor_format)
    if box is None:
        return image
    if (anchor_ref_w, anchor_ref_h) != (curr_w, curr_h):
        box = _rescale_box(box, anchor_ref_w, anchor_ref_h, curr_w, curr_h)
    if anchor_type == ANCHOR_TYPE_CROP:
        image = image.crop(box)
        if resize_after_crop:
            target_w, target_h = resize_target or (curr_w, curr_h)
            image = image.resize((target_w, target_h), Image.BICUBIC)
        return image
    if anchor_type == ANCHOR_TYPE_DRAW:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        line_width = max(2, min(image.width, image.height) // 200)
        draw.rectangle(_to_draw_box(box, image.width, image.height, line_width), outline='red', width=line_width)
    return image


def _load_image_flexible(image: Any) -> Image.Image:
    return Template._load_image(image, load_images=True)


@lru_cache(maxsize=1)
def _get_qwen_vl_vision_process():
    from qwen_vl_utils import vision_process
    return vision_process


def _get_qwen_vl_constant(name: str, default: Any) -> Any:
    vision_process = _get_qwen_vl_vision_process()
    return getattr(vision_process, name, default)


def _smart_resize_qwen(height: int,
                       width: int,
                       *,
                       factor: int,
                       min_pixels: Optional[int] = None,
                       max_pixels: Optional[int] = None) -> Tuple[int, int]:
    vision_process = _get_qwen_vl_vision_process()
    return vision_process.smart_resize(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == 'RGBA':
        white_background = Image.new('RGB', image.size, (255, 255, 255))
        white_background.paste(image, mask=image.split()[3])
        return white_background
    return image.convert('RGB')


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _frame_to_hwc(frame: np.ndarray, channel_last: bool) -> np.ndarray:
    if channel_last:
        return frame
    return np.transpose(frame, (1, 2, 0))


def _frame_from_hwc(frame_hwc: np.ndarray, channel_last: bool) -> np.ndarray:
    if channel_last:
        return frame_hwc
    return np.transpose(frame_hwc, (2, 0, 1))


def _frame_to_pil_image(frame: Any) -> Optional[Image.Image]:
    if isinstance(frame, Image.Image):
        return _to_rgb(frame.copy())

    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    elif isinstance(frame, np.ndarray):
        arr = frame
    else:
        try:
            return _to_rgb(_load_image_flexible(frame))
        except Exception:
            return None

    if arr.ndim != 3:
        return None
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    try:
        return _to_rgb(Image.fromarray(arr))
    except Exception:
        return None


def _video_last_frame_to_pil(video: Any) -> Optional[Image.Image]:
    if isinstance(video, torch.Tensor) and video.ndim == 4 and video.shape[0] > 0:
        return _frame_to_pil_image(video[-1])
    if isinstance(video, np.ndarray) and video.ndim == 4 and video.shape[0] > 0:
        frame = video[-1]
        if _is_channel_first_video(video):
            frame = np.transpose(frame, (1, 2, 0))
        return _frame_to_pil_image(frame)
    if isinstance(video, (list, tuple)) and video:
        return _frame_to_pil_image(video[-1])
    return None


def fetch_image_with_anchor(ele: Dict[str, Union[str, Image.Image]],
                            *,
                            anchor: Any = None,
                            anchor_type: int = 0,
                            shape_wh: Optional[Tuple[int, int]] = None,
                            image_patch_size: int = 14,
                            anchor_format: str = 'xyxy') -> Image.Image:
    image = ele['image'] if 'image' in ele else ele['image_url']
    image = _to_rgb(_load_image_flexible(image))

    # Crop should happen before resize so downstream resolution policy uses ROI.
    if anchor is not None and anchor_type == ANCHOR_TYPE_CROP:
        image = _apply_anchor_to_image_obj(
            image,
            anchor,
            anchor_type,
            shape_wh=shape_wh,
            resize_after_crop=False,
            anchor_format=anchor_format)

    patch_factor = int(image_patch_size * _get_qwen_vl_constant('SPATIAL_MERGE_SIZE', 2))
    if 'resized_height' in ele and 'resized_width' in ele:
        resized_height, resized_width = _smart_resize_qwen(
            ele['resized_height'], ele['resized_width'], factor=patch_factor)
    else:
        width, height = image.size
        min_pixels = ele.get('min_pixels', _get_qwen_vl_constant('IMAGE_MIN_TOKEN_NUM', 4) * patch_factor**2)
        max_pixels = ele.get('max_pixels', _get_qwen_vl_constant('IMAGE_MAX_TOKEN_NUM', 16384) * patch_factor**2)
        resized_height, resized_width = _smart_resize_qwen(
            height, width, factor=patch_factor, min_pixels=min_pixels, max_pixels=max_pixels)
    image = image.resize((resized_width, resized_height), Image.BICUBIC)

    # Draw mode follows post-resize path.
    if anchor is not None and anchor_type == ANCHOR_TYPE_DRAW:
        image = _apply_anchor_to_image_obj(
            image,
            anchor,
            anchor_type,
            shape_wh=shape_wh,
            resize_after_crop=False,
            anchor_format=anchor_format)
    return image


def _load_video_frames_as_tensor(video_frames: Any) -> torch.Tensor:
    assert isinstance(video_frames, (list, tuple)) and video_frames
    images = [_to_rgb(_load_image_flexible(frame)) for frame in video_frames]
    base_size = images[0].size
    tensors = []
    for image in images:
        if image.size != base_size:
            image = image.resize(base_size, Image.BICUBIC)
        tensors.append(torch.from_numpy(np.asarray(image).transpose(2, 0, 1)))
    return torch.stack(tensors)


def fetch_video_with_anchor(ele: Dict[str, Any],
                            *,
                            anchor: Any = None,
                            anchor_type: int = 0,
                            shape_wh: Optional[Tuple[int, int]] = None,
                            image_patch_size: int = 14,
                            return_video_sample_fps: bool = False,
                            return_video_metadata: bool = False,
                            anchor_format: str = 'xyxy') -> Any:
    image_factor = image_patch_size * _get_qwen_vl_constant('SPATIAL_MERGE_SIZE', 2)
    video_min_token_num = _get_qwen_vl_constant('VIDEO_MIN_TOKEN_NUM', 128)
    video_max_token_num = _get_qwen_vl_constant('VIDEO_MAX_TOKEN_NUM', 768)
    frame_factor = _get_qwen_vl_constant('FRAME_FACTOR', 2)
    model_seq_len = _get_qwen_vl_constant('MODEL_SEQ_LEN', int(float(os.environ.get('MODEL_SEQ_LEN', 128000))))

    video_frame_min_pixels = video_min_token_num * image_factor * image_factor
    video_frame_max_pixels = video_max_token_num * image_factor * image_factor

    if isinstance(ele['video'], str):
        vision_process = _get_qwen_vl_vision_process()
        backends = vision_process.VIDEO_READER_BACKENDS
        video_reader_backend = vision_process.get_video_reader_backend()
        try:
            video, video_metadata, sample_fps = backends[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f'video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}')
            video, video_metadata, sample_fps = backends['torchvision'](ele)
    else:
        video = _load_video_frames_as_tensor(ele['video'])
        nframes = _ceil_by_factor(len(video), frame_factor)
        if len(video) < nframes:
            pad = video[-1:].repeat(nframes - len(video), 1, 1, 1)
            video = torch.cat([video, pad], dim=0)
        sample_fps = ele.get('sample_fps', 2.0)
        raw_fps = ele.get('raw_fps', sample_fps)
        video_metadata = dict(
            fps=raw_fps,
            frames_indices=[i for i in range(len(video))],
            total_num_frames=(nframes / sample_fps) * raw_fps,
            video_backend='frame_list')

    # Crop mode should happen before resize.
    if anchor is not None and anchor_type == ANCHOR_TYPE_CROP:
        video = _apply_anchor_to_torch_video(
            video,
            anchor,
            anchor_type,
            shape_wh=shape_wh,
            resize_after_crop=False,
            anchor_format=anchor_format)

    nframes, _, height, width = video.shape
    min_pixels = ele.get('min_pixels', video_frame_min_pixels)
    total_pixels = ele.get('total_pixels', model_seq_len * image_factor * image_factor * 0.9)
    max_pixels = max(min(video_frame_max_pixels, total_pixels / max(nframes, 1) * frame_factor), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get('max_pixels', max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f'The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].')
    max_pixels = min(max_pixels_supposed, max_pixels)

    if 'resized_height' in ele and 'resized_width' in ele:
        resized_height, resized_width = _smart_resize_qwen(
            ele['resized_height'], ele['resized_width'], factor=image_factor)
    else:
        resized_height, resized_width = _smart_resize_qwen(
            height, width, factor=image_factor, min_pixels=min_pixels, max_pixels=max_pixels)

    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True).float()

    # Draw mode follows post-resize path.
    if anchor is not None and anchor_type == ANCHOR_TYPE_DRAW:
        video = _apply_anchor_to_torch_video(
            video,
            anchor,
            anchor_type,
            shape_wh=shape_wh,
            resize_after_crop=False,
            anchor_format=anchor_format)

    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


def _is_channel_last_video(arr: Any) -> bool:
    return arr.ndim == 4 and arr.shape[-1] in (1, 3, 4)


def _is_channel_first_video(arr: Any) -> bool:
    return arr.ndim == 4 and arr.shape[1] in (1, 3, 4)


def _apply_anchor_to_numpy_video(video: np.ndarray,
                                 anchor: Any,
                                 anchor_type: int,
                                 *,
                                 shape_wh: Optional[Tuple[int, int]] = None,
                                 resize_after_crop: bool = False,
                                 anchor_format: str = 'auto') -> np.ndarray:
    if video.ndim != 4:
        return video
    if anchor_type not in {ANCHOR_TYPE_CROP, ANCHOR_TYPE_DRAW}:
        return video
    channel_last = _is_channel_last_video(video)
    if not channel_last and not _is_channel_first_video(video):
        return video

    if channel_last:
        curr_h, curr_w = video.shape[1], video.shape[2]
    else:
        curr_h, curr_w = video.shape[2], video.shape[3]
    anchor_ref_w, anchor_ref_h = shape_wh or (curr_w, curr_h)
    box = _normalize_anchor(anchor, anchor_ref_w, anchor_ref_h, anchor_format=anchor_format)
    if box is None:
        return video
    if (anchor_ref_w, anchor_ref_h) != (curr_w, curr_h):
        box = _rescale_box(box, anchor_ref_w, anchor_ref_h, curr_w, curr_h)
    x1, y1, x2, y2 = box

    if anchor_type == ANCHOR_TYPE_CROP:
        if channel_last:
            cropped = video[:, y1:y2, x1:x2, ...]
        else:
            cropped = video[:, :, y1:y2, x1:x2]
        # Keep compatibility for callers that expect original resolution after crop.
        if not resize_after_crop:
            return np.ascontiguousarray(cropped)
        out = []
        for frame in cropped:
            frame_hwc = _frame_to_hwc(frame, channel_last)
            img = Image.fromarray(np.asarray(frame_hwc, dtype=np.uint8))
            img = img.resize((curr_w, curr_h), Image.BICUBIC)
            frame_out = np.asarray(img)
            out.append(_frame_from_hwc(frame_out, channel_last))
        return np.ascontiguousarray(np.stack(out, axis=0))

    out = []
    for frame in video:
        frame_hwc = _frame_to_hwc(frame, channel_last)
        frame_hwc = np.asarray(frame_hwc, dtype=np.uint8)
        img = Image.fromarray(frame_hwc).copy()
        draw = ImageDraw.Draw(img)
        line_width = max(2, min(curr_w, curr_h) // 200)
        draw_box = _to_draw_box((x1, y1, x2, y2), curr_w, curr_h, line_width)
        draw.rectangle(draw_box, outline='red', width=line_width)
        out.append(_frame_from_hwc(np.asarray(img), channel_last))
    return np.ascontiguousarray(np.stack(out, axis=0))


def _apply_anchor_to_torch_video(video: torch.Tensor,
                                 anchor: Any,
                                 anchor_type: int,
                                 *,
                                 shape_wh: Optional[Tuple[int, int]] = None,
                                 resize_after_crop: bool = False,
                                 anchor_format: str = 'auto') -> torch.Tensor:
    if video.ndim != 4:
        return video
    if anchor_type not in {ANCHOR_TYPE_CROP, ANCHOR_TYPE_DRAW}:
        return video
    video_np = video.detach().cpu().numpy()
    out_np = _apply_anchor_to_numpy_video(
        video_np,
        anchor,
        anchor_type,
        shape_wh=shape_wh,
        resize_after_crop=resize_after_crop,
        anchor_format=anchor_format)
    out = torch.from_numpy(out_np).to(video.device)
    if out.dtype != video.dtype:
        out = out.to(video.dtype)
    return out


def _apply_anchor_to_video(video: Any,
                           anchor: Any,
                           anchor_type: int,
                           *,
                           shape_wh: Optional[Tuple[int, int]] = None,
                           resize_after_crop: bool = False,
                           anchor_format: str = 'auto') -> Any:
    if anchor_type == ANCHOR_TYPE_NONE:
        return video
    if isinstance(video, torch.Tensor):
        return _apply_anchor_to_torch_video(
            video,
            anchor,
            anchor_type,
            shape_wh=shape_wh,
            resize_after_crop=resize_after_crop,
            anchor_format=anchor_format)
    if isinstance(video, np.ndarray):
        return _apply_anchor_to_numpy_video(
            video,
            anchor,
            anchor_type,
            shape_wh=shape_wh,
            resize_after_crop=resize_after_crop,
            anchor_format=anchor_format)
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
                    resize_after_crop=resize_after_crop and (anchor_type == ANCHOR_TYPE_CROP),
                    resize_target=shape_wh or (img.width, img.height),
                    anchor_format=anchor_format)
                processed.append(np.asarray(img))
            except Exception:
                processed.append(frame)
        return processed
    return video


class Qwen3VLAnchorTemplate(Qwen3VLTemplate):

    @staticmethod
    def _get_shape_wh(inputs, media_type: str, index: int) -> Optional[Tuple[int, int]]:
        # Explicit keys requested by user.
        if media_type == 'image':
            shape_raw = _pick_media_value(inputs.extra_kwargs.get('image_shape'), media_type, index)
        else:
            shape_raw = _pick_media_value(inputs.extra_kwargs.get('video_shape'), media_type, index)
        if shape_raw is None:
            # backward compatibility fallback
            shape_raw = _pick_media_value(inputs.extra_kwargs.get('shape'), media_type, index)
        return _parse_shape_wh(shape_raw)

    @staticmethod
    def _collect_anchor_info(media_type: str, index: int, inputs) -> Tuple[Any, int, Optional[Tuple[int, int]], str]:
        anchor_type_raw = _pick_media_value(inputs.extra_kwargs.get('anchor_type', ANCHOR_TYPE_NONE), media_type, index)
        anchor_type = _normalize_anchor_type(anchor_type_raw)
        anchor = _pick_media_value(inputs.extra_kwargs.get('anchors'), media_type, index)
        # [0,0,0,0] means "no crop", but draw mode should still draw it.
        if anchor_type == ANCHOR_TYPE_CROP and _is_noop_anchor(anchor):
            anchor = None
            anchor_type = ANCHOR_TYPE_NONE
        shape_wh = Qwen3VLAnchorTemplate._get_shape_wh(inputs, media_type, index)
        return anchor, anchor_type, shape_wh, 'xyxy'

    @staticmethod
    def _append_anchor_kwargs(media_type: str, inputs, anchor: Any, anchor_type: int) -> None:
        if anchor is None:
            return
        inputs.mm_processor_kwargs.setdefault(f'{media_type}_anchors', []).append(anchor)
        inputs.mm_processor_kwargs.setdefault(f'{media_type}_anchor_type', anchor_type)

    def _build_fetch_kwargs(self, inputs) -> Dict[str, Any]:
        kwargs = {'image_patch_size': self.processor.image_processor.patch_size} if self.version == 'v3' else {}
        if self.mode == 'vllm':
            # resized in qwen_vl_utils, no need to resize again in vllm
            inputs.mm_processor_kwargs['do_resize'] = False
        return kwargs

    @staticmethod
    def _is_debug_save_enabled(inputs) -> bool:
        flag = inputs.extra_kwargs.get('save_anchor_preview', None)
        if flag is None:
            flag = inputs.extra_kwargs.get('anchor_debug_save', False)
        return _to_bool(flag, default=False)

    @staticmethod
    def _debug_save_dir(inputs) -> str:
        save_dir = _json_loads_maybe(inputs.extra_kwargs.get('anchor_debug_dir', 'anchor_debug_outputs'))
        if not isinstance(save_dir, str) or not save_dir.strip():
            return 'anchor_debug_outputs'
        return save_dir.strip()

    def _save_media_preview(self, media_type: str, index: int, media: Any, inputs, anchor_type: int) -> None:
        if not self._is_debug_save_enabled(inputs):
            return
        save_dir = self._debug_save_dir(inputs)
        os.makedirs(save_dir, exist_ok=True)
        if media_type == 'image':
            image = _frame_to_pil_image(media)
        else:
            image = _video_last_frame_to_pil(media)
        if image is None:
            logger.warning_once('Failed to build debug preview image from media; skip saving.')
            return
        ts = int(time.time() * 1000)
        filename = f'{media_type}_{index}_{_anchor_type_name(anchor_type)}_{ts}.png'
        path = os.path.join(save_dir, filename)
        image.save(path)
        logger.info(f'Anchor debug preview saved to: {path}')

    def _replace_image_tag(self, index: int, inputs, anchor: Any, anchor_type: int, shape_wh, anchor_format: str,
                           fetch_kwargs: Dict[str, Any]):
        image_ele = {'image': inputs.images[index]}
        if _has_anchor_operation(anchor, anchor_type):
            inputs.images[index] = fetch_image_with_anchor(
                image_ele,
                anchor=anchor,
                anchor_type=anchor_type,
                shape_wh=shape_wh,
                anchor_format=anchor_format,
                **fetch_kwargs)
        else:
            from qwen_vl_utils import fetch_image
            inputs.images[index] = fetch_image(image_ele, **fetch_kwargs)
        self._save_media_preview('image', index, inputs.images[index], inputs, anchor_type)
        if self.mode == 'lmdeploy':
            return ['<|vision_start|>', [-100], '<|vision_end|>']
        return ['<|vision_start|><|image_pad|><|vision_end|>']

    def _replace_video_tag(self, index: int, inputs, anchor: Any, anchor_type: int, shape_wh, anchor_format: str,
                           fetch_kwargs: Dict[str, Any]):
        video_fetch_kwargs = dict(fetch_kwargs)
        if self.version == 'v3':
            video_fetch_kwargs['return_video_metadata'] = True
        video = inputs.videos[index]
        video_inputs = {'video': video}
        if isinstance(video, list):  # image list
            video_inputs['sample_fps'] = _get_qwen_vl_constant('FPS', 2.0)

        if _has_anchor_operation(anchor, anchor_type):
            video, video_kwargs = fetch_video_with_anchor(
                video_inputs,
                return_video_sample_fps=True,
                anchor=anchor,
                anchor_type=anchor_type,
                shape_wh=shape_wh,
                anchor_format=anchor_format,
                **video_fetch_kwargs)
        else:
            from qwen_vl_utils import fetch_video
            video, video_kwargs = fetch_video(video_inputs, return_video_sample_fps=True, **video_fetch_kwargs)

        tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
        if self.version == 'v2_5':
            inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
        elif self.version == 'v3':
            if self.mode != 'vllm':
                video, video_metadata = video
                inputs.mm_processor_kwargs.setdefault('video_metadata', []).append(video_metadata)
                tokens = ['<|video_pad|>']
            inputs.mm_processor_kwargs['do_sample_frames'] = False
        if isinstance(video, torch.Tensor):
            video = video.to(torch.uint8)
        inputs.videos[index] = video
        self._save_media_preview('video', index, inputs.videos[index], inputs, anchor_type)
        return tokens

    def replace_tag(self, media_type, index, inputs):
        if media_type not in {'image', 'video'}:
            return super().replace_tag(media_type, index, inputs)
        anchor, anchor_type, shape_wh, anchor_format = self._collect_anchor_info(media_type, index, inputs)
        self._append_anchor_kwargs(media_type, inputs, anchor, anchor_type)
        fetch_kwargs = self._build_fetch_kwargs(inputs)
        if media_type == 'image':
            return self._replace_image_tag(index, inputs, anchor, anchor_type, shape_wh, anchor_format, fetch_kwargs)
        return self._replace_video_tag(index, inputs, anchor, anchor_type, shape_wh, anchor_format, fetch_kwargs)

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

