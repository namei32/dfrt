from __future__ import annotations

from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class DefectType(str, Enum):
    """Coarse procedural mask families kept for backward-compatible imports."""

    SCRATCH = "scratch"
    CRACK = "crack"
    PIT = "pit"
    BLOB = "blob"
    PATCH = "patch"


def create_composite_preserving_background(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Composite generated pixels under mask while preserving the source context."""

    base = original.convert("RGB")
    edited = generated.resize(base.size, Image.Resampling.LANCZOS).convert("RGB")
    edit_mask = mask.resize(base.size, Image.Resampling.LANCZOS).convert("L")
    return Image.composite(edited, base, edit_mask)


def build_core_shell_masks(
    mask: Image.Image,
    *,
    core_threshold: int = 128,
    shell_radius: int = 3,
) -> tuple[Image.Image, Image.Image]:
    """Return hard core and boundary shell masks for a soft edit mask."""

    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    core = (arr >= int(core_threshold)).astype(np.uint8)
    radius = max(1, int(shell_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(core, kernel, iterations=1)
    eroded = cv2.erode(core, kernel, iterations=1)
    shell = np.clip(dilated - eroded, 0, 1).astype(np.uint8)
    return (
        Image.fromarray(core * 255, mode="L"),
        Image.fromarray(shell * 255, mode="L").filter(ImageFilter.GaussianBlur(radius=0.6)),
    )


def dilate_binary_mask(mask: np.ndarray | Image.Image, iterations: int = 1, *, kernel_size: int = 3) -> np.ndarray:
    """Dilate a binary mask and return a uint8 0/1 array."""

    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(mask, dtype=np.uint8)
    binary = (arr > 0).astype(np.uint8)
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(binary, kernel, iterations=max(0, int(iterations))).astype(np.uint8)


def generate_defect_mask(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    defect_type: Optional[DefectType | str] = None,
    seed: int = 0,
    feather_radius: float = 1.2,
) -> Image.Image:
    """Generate a simple procedural mask inside a bounding box.

    The retained DRFT-v2 path uses morphology-calibrated masks, but older code
    imports this helper. Keep it deterministic and conservative.
    """

    width, height = size
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
    y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
    mask = Image.new("L", (width, height), 0)
    if x1 <= x0 or y1 <= y0:
        return mask

    rng = np.random.default_rng(int(seed))
    draw = ImageDraw.Draw(mask)
    family = str(defect_type.value if isinstance(defect_type, DefectType) else defect_type or "")
    if family in {"scratch", "crack"}:
        points = []
        steps = max(3, min(9, (y1 - y0 + x1 - x0) // 16))
        for idx in range(steps):
            t = idx / max(1, steps - 1)
            x = int(round(x0 + t * (x1 - x0) + rng.normal(0, max(1.0, (x1 - x0) * 0.06))))
            y = int(round(y0 + t * (y1 - y0) + rng.normal(0, max(1.0, (y1 - y0) * 0.06))))
            points.append((max(x0, min(x1, x)), max(y0, min(y1, y))))
        draw.line(points, fill=255, width=max(1, min(x1 - x0, y1 - y0) // 5))
    elif family in {"pit", "blob"}:
        draw.ellipse((x0, y0, x1, y1), fill=255)
    else:
        inset_x = int((x1 - x0) * 0.08)
        inset_y = int((y1 - y0) * 0.08)
        draw.rounded_rectangle((x0 + inset_x, y0 + inset_y, x1 - inset_x, y1 - inset_y), radius=2, fill=255)

    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
    return mask
