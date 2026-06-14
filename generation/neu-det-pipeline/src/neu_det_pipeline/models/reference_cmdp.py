from __future__ import annotations

import json
import math
import re
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data.loader import DefectSample
from ..guidance.mask import create_composite_preserving_background


def slugify_class_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip())
    return slug.strip("_") or "unknown"


def infer_surface_domain(dataset_root: Path | str, explicit: str = "auto") -> str:
    value = explicit.lower().strip()
    if value in {"steel", "fabric"}:
        return value
    root_text = str(dataset_root).lower()
    if any(key in root_text for key in ("tilda", "mvtec", "fabric", "textile")):
        return "fabric"
    return "steel"


CLASS_AWARE_DEFECT_DESCRIPTORS = {
    "crazing": "fine interconnected crazing cracks",
    "inclusion": "dark embedded inclusion particles",
    "patches": "irregular oxidized surface patches",
    "pitted_surface": "clustered pitted surface cavities",
    "rolled-in_scale": "rolled-in oxide scale streaks",
    "rolled_in_scale": "rolled-in oxide scale streaks",
    "scratches": "thin linear scratch grooves",
    "punching_hole": "dark round punching hole defects",
    "welding_line": "vertical welding line seam defects",
    "crescent_gap": "crescent-shaped edge gap defects",
    "water_spot": "faint water spot stains",
    "oil_spot": "elliptical oil spot contamination",
    "rolled_pit": "clustered rolled pit marks",
    "silk_spot": "fine silk-like spot streaks",
    "crease": "long shallow crease lines",
    "waist_folding": "subtle waist folding deformation lines",
}


def reference_caption(domain: str, class_name: Optional[str] = None) -> str:
    surface = "fabric" if domain.lower() == "fabric" else "hot-rolled steel strip"
    base = f"macro grayscale inspection photo of a defect on the {surface} surface"
    if not class_name:
        return base

    normalized = slugify_class_name(str(class_name)).lower()
    descriptor = CLASS_AWARE_DEFECT_DESCRIPTORS.get(normalized)
    if descriptor is None:
        descriptor = str(class_name).replace("_", " ").replace("-", " ").strip() or "surface defect"
    return f"{base}, {descriptor}, realistic industrial texture"


def clip_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def expand_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    dilation: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = clip_bbox(bbox, size)
    width = max(1.0, float(x1 - x0))
    height = max(1.0, float(y1 - y0))
    scale = max(1.0, float(dilation))
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    new_w = width * scale
    new_h = height * scale
    return clip_bbox(
        (
            int(round(cx - new_w * 0.5)),
            int(round(cy - new_h * 0.5)),
            int(round(cx + new_w * 0.5)),
            int(round(cy + new_h * 0.5)),
        ),
        size,
    )


def bbox_inside_crop(
    bbox: tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    cx0, cy0, _cx1, _cy1 = crop_bbox
    return x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0


def scale_bbox_to_size(
    bbox: tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_w, src_h = source_size
    dst_w, dst_h = target_size
    x0, y0, x1, y1 = bbox
    return clip_bbox(
        (
            int(round(x0 * dst_w / max(1, src_w))),
            int(round(y0 * dst_h / max(1, src_h))),
            int(round(x1 * dst_w / max(1, src_w))),
            int(round(y1 * dst_h / max(1, src_h))),
        ),
        target_size,
    )


def build_scmp_pixel_masks(
    crop_size: tuple[int, int],
    defect_bbox_in_crop: tuple[int, int, int, int],
    *,
    feather_radius: float = 2.0,
) -> tuple[Image.Image, Image.Image]:
    """Return paper SCMP masks.

    The context mask M is 1 outside the original defect box and 0 inside
    it. The defect mask is 1 inside the original defect box and 0 outside.
    """

    width, height = crop_size
    bbox = clip_bbox(defect_bbox_in_crop, crop_size)
    defect = Image.new("L", crop_size, 0)
    draw = ImageDraw.Draw(defect)
    draw.rectangle(bbox, fill=255)
    if feather_radius > 0:
        defect = defect.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    defect_np = np.asarray(defect.convert("L"), dtype=np.uint8)
    context_np = (255 - defect_np).astype(np.uint8)
    return Image.fromarray(context_np, mode="L"), Image.fromarray(defect_np, mode="L")


def _binary_mask(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L"), dtype=np.uint8) > 127


def _mask_from_binary(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _dilate_mask(mask: Image.Image, radius: int) -> Image.Image:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.convert("L")
    return mask.convert("L").filter(ImageFilter.MaxFilter(radius * 2 + 1))


def _erode_mask(mask: Image.Image, radius: int) -> Image.Image:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.convert("L")
    return mask.convert("L").filter(ImageFilter.MinFilter(radius * 2 + 1))


def _otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.5
    if float(values.max() - values.min()) < 1e-6:
        return float(values.mean())
    hist, bin_edges = np.histogram(values, bins=128, range=(0.0, 1.0))
    total = float(hist.sum())
    if total <= 0:
        return 0.5
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    sum_total = float((hist * centers).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_threshold = 0.5
    for count, center in zip(hist, centers):
        weight_bg += float(count)
        if weight_bg <= 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0:
            break
        sum_bg += float(count) * float(center)
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            best_threshold = float(center)
    return best_threshold


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 95))
    if hi - lo < 1e-6:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi - lo < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(-1)
    if flat.size == 0:
        return np.zeros_like(values, dtype=bool)
    fraction = min(1.0, max(0.0, float(fraction)))
    k = max(1, int(round(flat.size * fraction)))
    k = min(k, flat.size)
    selected = np.argpartition(flat, flat.size - k)[flat.size - k :]
    mask = np.zeros(flat.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(values.shape)


def estimate_pseudo_defect_mask(
    image: Image.Image,
    defect_bbox_in_crop: tuple[int, int, int, int],
    *,
    blur_sigma: float = 2.0,
    gradient_weight: float = 0.5,
    tau: Optional[float] = None,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.25,
) -> Image.Image:
    """Estimate the DB-SCMP pseudo defect contour P inside a bbox.

    The residual follows the proposal:
    |I_B - G_sigma(I_B)| + lambda_g |grad(I_B) - grad(G_sigma(I_B))|.
    When tau is not supplied, an Otsu threshold on the normalized residual is
    used, with area guards so tiny or all-bbox masks do not collapse DBT.
    """

    image = image.convert("RGB")
    bbox = clip_bbox(defect_bbox_in_crop, image.size)
    roi = image.crop(bbox)
    if roi.size[0] <= 1 or roi.size[1] <= 1:
        fallback = Image.new("L", image.size, 0)
        ImageDraw.Draw(fallback).rectangle(bbox, fill=255)
        return fallback

    bg = roi.filter(ImageFilter.GaussianBlur(radius=max(0.1, float(blur_sigma))))
    roi_arr = np.asarray(roi, dtype=np.float32) / 255.0
    bg_arr = np.asarray(bg, dtype=np.float32) / 255.0
    intensity_residual = np.mean(np.abs(roi_arr - bg_arr), axis=2)

    roi_gray = roi_arr.mean(axis=2)
    bg_gray = bg_arr.mean(axis=2)
    roi_gy, roi_gx = np.gradient(roi_gray)
    bg_gy, bg_gx = np.gradient(bg_gray)
    gradient_residual = np.hypot(roi_gx - bg_gx, roi_gy - bg_gy)
    residual = intensity_residual + max(0.0, float(gradient_weight)) * gradient_residual
    residual = _robust_normalize(residual)

    if tau is None:
        threshold = _otsu_threshold(residual)
    else:
        threshold = float(tau)
        if threshold > 1.0:
            threshold = threshold / 255.0
        threshold = min(1.0, max(0.0, threshold))

    mask = residual > threshold
    area_ratio = float(mask.mean()) if mask.size else 0.0
    min_area_ratio = min(0.5, max(0.0, float(min_area_ratio)))
    max_area_ratio = min(0.95, max(min_area_ratio + 1e-3, float(max_area_ratio)))
    if area_ratio < min_area_ratio:
        mask = _top_fraction_mask(residual, min_area_ratio)
    elif area_ratio > max_area_ratio:
        mask = _top_fraction_mask(residual, max_area_ratio)

    if not mask.any():
        y, x = np.unravel_index(int(np.argmax(residual)), residual.shape)
        mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = True

    mask_img = _mask_from_binary(mask)
    # Close small gaps while preserving thin scratches/crazing better than an
    # opening operation would.
    mask_img = _erode_mask(_dilate_mask(mask_img, 1), 1)
    closed = _binary_mask(mask_img)
    if float(closed.mean()) > max_area_ratio:
        mask_img = _mask_from_binary(_top_fraction_mask(residual, max_area_ratio))
    full = Image.new("L", image.size, 0)
    full.paste(mask_img, bbox[:2])
    return full


@dataclass
class DBTMaskSet:
    pseudo: Image.Image
    core: Image.Image
    band: Image.Image
    out: Image.Image
    defect: Image.Image
    stats: dict[str, float | int | bool | str]


@dataclass(frozen=True)
class StochasticBoxParams:
    band_radius: int
    residual_blur_sigma: float
    gradient_weight: float
    max_area_ratio: float
    box_inner_strength: float
    box_pseudo_strength: float
    box_boundary_strength: float
    family: str


def _stable_sample_seed(base_seed: int, cls_name: str, target_key: str, index: int) -> int:
    payload = f"{int(base_seed)}|{cls_name}|{target_key}|{int(index)}".encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0xFFFFFFFF


def _sample_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    if high < low:
        low, high = high, low
    if math.isclose(low, high):
        return low
    return float(rng.uniform(low, high))


def _defect_family(cls_name: str) -> str:
    cls = slugify_class_name(cls_name).lower()
    if any(key in cls for key in ("crazing", "scratch", "crease", "silk_spot", "welding_line", "waist_folding")):
        return "linear"
    if any(key in cls for key in ("punching_hole", "crescent_gap")):
        return "hole"
    if any(key in cls for key in ("oil_spot", "water_spot", "patch", "inclusion", "pitted", "rolled_pit")):
        return "blob"
    return "default"


def sample_stochastic_box_params(
    cls_name: str,
    target_key: str,
    *,
    index: int,
    seed: int,
    profile: str = "balanced",
    base_band_radius: int = 8,
    base_residual_blur_sigma: float = 2.0,
    base_gradient_weight: float = 0.5,
    base_max_area_ratio: float = 0.30,
    base_box_inner_strength: float = 0.85,
    base_box_pseudo_strength: float = 1.0,
    base_box_boundary_strength: float = 0.45,
) -> StochasticBoxParams:
    """Sample reproducible per-instance box-open parameters.

    The balanced profile opens the bbox enough to create variants while keeping
    boundary-sensitive classes more conservative for label safety.
    """

    family = _defect_family(cls_name)
    ranges = {
        "linear": {
            "band_radius": (5, 10),
            "residual_blur_sigma": (1.4, 2.6),
            "gradient_weight": (0.45, 0.75),
            "max_area_ratio": (0.18, 0.30),
            "box_inner_strength": (0.72, 0.90),
            "box_pseudo_strength": (0.96, 1.0),
            "box_boundary_strength": (0.45, 0.65),
        },
        "blob": {
            "band_radius": (6, 12),
            "residual_blur_sigma": (1.6, 3.2),
            "gradient_weight": (0.30, 0.65),
            "max_area_ratio": (0.24, 0.38),
            "box_inner_strength": (0.80, 0.98),
            "box_pseudo_strength": (0.96, 1.0),
            "box_boundary_strength": (0.35, 0.55),
        },
        "hole": {
            "band_radius": (5, 10),
            "residual_blur_sigma": (1.5, 2.8),
            "gradient_weight": (0.40, 0.70),
            "max_area_ratio": (0.20, 0.32),
            "box_inner_strength": (0.78, 0.95),
            "box_pseudo_strength": (0.96, 1.0),
            "box_boundary_strength": (0.45, 0.65),
        },
        "default": {
            "band_radius": (5, 12),
            "residual_blur_sigma": (1.5, 3.0),
            "gradient_weight": (0.35, 0.70),
            "max_area_ratio": (0.22, 0.35),
            "box_inner_strength": (0.75, 0.95),
            "box_pseudo_strength": (0.95, 1.0),
            "box_boundary_strength": (0.35, 0.60),
        },
    }
    selected = ranges.get(family, ranges["default"])
    normalized_profile = str(profile).lower().replace("_", "-")
    if normalized_profile in {"off", "deterministic", "none"}:
        return StochasticBoxParams(
            band_radius=int(base_band_radius),
            residual_blur_sigma=float(base_residual_blur_sigma),
            gradient_weight=float(base_gradient_weight),
            max_area_ratio=float(base_max_area_ratio),
            box_inner_strength=float(base_box_inner_strength),
            box_pseudo_strength=float(base_box_pseudo_strength),
            box_boundary_strength=float(base_box_boundary_strength),
            family=family,
        )
    if normalized_profile in {"open", "aggressive"}:
        selected = {
            **selected,
            "box_inner_strength": (min(0.98, selected["box_inner_strength"][0] + 0.05), 1.0),
            "box_boundary_strength": (max(0.25, selected["box_boundary_strength"][0] - 0.08), selected["box_boundary_strength"][1]),
            "max_area_ratio": (selected["max_area_ratio"][0], min(0.45, selected["max_area_ratio"][1] + 0.05)),
        }
    elif normalized_profile in {"safe", "conservative"}:
        selected = {
            **selected,
            "box_inner_strength": (selected["box_inner_strength"][0], min(selected["box_inner_strength"][1], 0.90)),
            "box_boundary_strength": (max(0.45, selected["box_boundary_strength"][0]), selected["box_boundary_strength"][1]),
            "max_area_ratio": (selected["max_area_ratio"][0], min(selected["max_area_ratio"][1], 0.30)),
        }

    rng = np.random.default_rng(_stable_sample_seed(seed, cls_name, target_key, index))
    band_min, band_max = selected["band_radius"]
    return StochasticBoxParams(
        band_radius=int(rng.integers(int(band_min), int(band_max) + 1)),
        residual_blur_sigma=_sample_uniform(rng, selected["residual_blur_sigma"]),
        gradient_weight=_sample_uniform(rng, selected["gradient_weight"]),
        max_area_ratio=_sample_uniform(rng, selected["max_area_ratio"]),
        box_inner_strength=_sample_uniform(rng, selected["box_inner_strength"]),
        box_pseudo_strength=_sample_uniform(rng, selected["box_pseudo_strength"]),
        box_boundary_strength=_sample_uniform(rng, selected["box_boundary_strength"]),
        family=family,
    )


def build_dbt_pixel_masks(
    image: Image.Image,
    defect_bbox_in_crop: tuple[int, int, int, int],
    *,
    pseudo_mask: Optional[Image.Image] = None,
    residual_blur_sigma: float = 2.0,
    gradient_weight: float = 0.5,
    tau: Optional[float] = None,
    core_radius: int = 2,
    band_radius: int = 8,
    small_core_ratio: float = 0.2,
    small_defect_band_radius: int = 1,
    out_safety_margin: int = 0,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.25,
) -> DBTMaskSet:
    """Build DB-SCMP pseudo contour tri-domain masks: core, band, out."""

    image = image.convert("RGB")
    if pseudo_mask is None:
        pseudo_mask = estimate_pseudo_defect_mask(
            image,
            defect_bbox_in_crop,
            blur_sigma=residual_blur_sigma,
            gradient_weight=gradient_weight,
            tau=tau,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
    else:
        pseudo_mask = pseudo_mask.convert("L").resize(image.size, Image.Resampling.NEAREST)

    pseudo = _binary_mask(pseudo_mask)
    if not pseudo.any():
        pseudo_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(pseudo_mask).rectangle(clip_bbox(defect_bbox_in_crop, image.size), fill=255)
        pseudo = _binary_mask(pseudo_mask)

    core_img = _erode_mask(pseudo_mask, core_radius)
    core = _binary_mask(core_img)
    pseudo_area = int(pseudo.sum())
    core_area = int(core.sum())
    small_core_protected = core_area < float(small_core_ratio) * max(1, pseudo_area)
    if small_core_protected:
        core = pseudo.copy()
        outer_radius = max(1, int(small_defect_band_radius))
    else:
        outer_radius = max(1, int(band_radius) + max(0, int(out_safety_margin)))

    outer = _binary_mask(_dilate_mask(pseudo_mask, outer_radius))
    band = outer & ~core
    out = ~outer
    defect = core | band

    total = max(1, image.size[0] * image.size[1])
    return DBTMaskSet(
        pseudo=_mask_from_binary(pseudo),
        core=_mask_from_binary(core),
        band=_mask_from_binary(band),
        out=_mask_from_binary(out),
        defect=_mask_from_binary(defect),
        stats={
            "pseudo_area": pseudo_area,
            "pseudo_area_ratio": float(pseudo_area / total),
            "core_area": int(core.sum()),
            "core_area_ratio": float(core.sum() / total),
            "band_area": int(band.sum()),
            "band_area_ratio": float(band.sum() / total),
            "out_area": int(out.sum()),
            "out_area_ratio": float(out.sum() / total),
            "outer_radius": int(outer_radius),
            "small_core_protected": bool(small_core_protected),
        },
    )


def build_box_adaptive_pixel_masks(
    image: Image.Image,
    defect_bbox_in_crop: tuple[int, int, int, int],
    *,
    pseudo_mask: Optional[Image.Image] = None,
    residual_blur_sigma: float = 2.0,
    gradient_weight: float = 0.5,
    tau: Optional[float] = None,
    band_radius: int = 8,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.30,
) -> DBTMaskSet:
    """Build box-level adaptive masks.

    The whole defect bbox is the editable defect domain. The pseudo mask is
    retained as a soft prior for generation strength, rather than a hard edit
    contour.
    """

    image = image.convert("RGB")
    bbox = clip_bbox(defect_bbox_in_crop, image.size)
    if pseudo_mask is None:
        pseudo_mask = estimate_pseudo_defect_mask(
            image,
            bbox,
            blur_sigma=residual_blur_sigma,
            gradient_weight=gradient_weight,
            tau=tau,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
    else:
        pseudo_mask = pseudo_mask.convert("L").resize(image.size, Image.Resampling.NEAREST)

    box_img = Image.new("L", image.size, 0)
    ImageDraw.Draw(box_img).rectangle(bbox, fill=255)
    box = _binary_mask(box_img)
    box_w = max(1, bbox[2] - bbox[0])
    box_h = max(1, bbox[3] - bbox[1])
    radius = max(1, min(int(band_radius), max(1, min(box_w, box_h) // 3)))

    core_img = _erode_mask(box_img, radius)
    core = _binary_mask(core_img)
    small_box_protected = not core.any()
    if small_box_protected:
        core = box.copy()
        band = np.zeros_like(box, dtype=bool)
    else:
        band = box & ~core
    out = ~box
    pseudo = _binary_mask(pseudo_mask) & box
    defect = box

    total = max(1, image.size[0] * image.size[1])
    return DBTMaskSet(
        pseudo=_mask_from_binary(pseudo),
        core=_mask_from_binary(core),
        band=_mask_from_binary(band),
        out=_mask_from_binary(out),
        defect=_mask_from_binary(defect),
        stats={
            "mask_geometry": "box-adaptive",
            "pseudo_area": int(pseudo.sum()),
            "pseudo_area_ratio": float(pseudo.sum() / total),
            "box_area": int(box.sum()),
            "box_area_ratio": float(box.sum() / total),
            "core_area": int(core.sum()),
            "core_area_ratio": float(core.sum() / total),
            "band_area": int(band.sum()),
            "band_area_ratio": float(band.sum() / total),
            "out_area": int(out.sum()),
            "out_area_ratio": float(out.sum() / total),
            "outer_radius": int(radius),
            "small_core_protected": bool(small_box_protected),
        },
    )


def mask_to_latent_tensor(mask: Image.Image, latent_hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    latent_h, latent_w = latent_hw
    arr = np.asarray(mask.convert("L").resize((latent_w, latent_h), Image.Resampling.BILINEAR), dtype=np.float32)
    arr = np.clip(arr / 255.0, 0.0, 1.0)
    return torch.from_numpy(arr)[None, None].to(device=device, dtype=dtype)


def normalize_latent_triplet(
    core: torch.Tensor,
    band: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = (core + band + out).clamp_min(float(eps))
    return core / total, band / total, out / total


def shape_preserving_pixel_blend(
    original_crop: Image.Image,
    generated_crop: Image.Image,
    core_mask: Image.Image,
    band_mask: Image.Image,
    *,
    core_strength: float = 1.0,
    band_strength: float = 0.55,
    residual_clip: Optional[float] = None,
    luminance_only: bool = False,
    contrast_preservation: float = 0.0,
    contrast_blur_sigma: float = 3.0,
    contrast_min: float = 2.5,
) -> Image.Image:
    """Blend generated residuals back through the pseudo-defect shape."""

    original = np.asarray(original_crop.convert("RGB"), dtype=np.float32)
    generated = np.asarray(generated_crop.resize(original_crop.size).convert("RGB"), dtype=np.float32)
    residual = generated - original
    if luminance_only:
        residual = residual.mean(axis=2, keepdims=True)
    if residual_clip is not None and float(residual_clip) > 0:
        clip = float(residual_clip)
        residual = np.clip(residual, -clip, clip)

    core = np.asarray(core_mask.resize(original_crop.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    band = np.asarray(band_mask.resize(original_crop.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    alpha = np.clip(
        core * max(0.0, float(core_strength)) + band * max(0.0, float(band_strength)),
        0.0,
        1.0,
    )
    contrast_preservation = min(1.0, max(0.0, float(contrast_preservation)))
    if contrast_preservation > 0:
        gray = original.mean(axis=2)
        local_bg = np.asarray(
            original_crop.convert("L").filter(ImageFilter.GaussianBlur(radius=max(0.1, float(contrast_blur_sigma)))),
            dtype=np.float32,
        )
        contrast = gray - local_bg
        residual_gray = residual.mean(axis=2) if residual.shape[2] == 3 else residual[..., 0]
        sign_ok = (np.abs(contrast) < float(contrast_min)) | (contrast * residual_gray >= 0.0)
        alpha *= np.where(sign_ok, 1.0, 1.0 - contrast_preservation).astype(np.float32)
    alpha = alpha[..., None]
    blended = np.clip(original + residual * alpha, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _gaussian_kernel_2d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = max(1e-3, float(sigma))
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel[None, None]


def propagate_background_latents(
    latents: torch.Tensor,
    out_mask: torch.Tensor,
    *,
    sigma: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """No-leak Gaussian propagation from M_out only."""

    sigma = float(sigma)
    if sigma <= 0:
        return latents
    kernel = _gaussian_kernel_2d(sigma, latents.device, latents.dtype)
    pad = kernel.shape[-1] // 2
    channels = latents.shape[1]
    conv_kernel = kernel.expand(channels, 1, -1, -1)
    pad_mode = "reflect" if latents.shape[-2] > pad and latents.shape[-1] > pad else "replicate"
    padded_num = F.pad(latents * out_mask, (pad, pad, pad, pad), mode=pad_mode)
    numerator = F.conv2d(padded_num, conv_kernel, groups=channels)
    padded_den = F.pad(out_mask, (pad, pad, pad, pad), mode=pad_mode)
    denominator = F.conv2d(padded_den, kernel).clamp_min(float(eps))
    return numerator / denominator


def transition_eta(
    scheduler: Any,
    timestep: torch.Tensor,
    *,
    eta_min: float,
    eta_max: float,
    step_idx: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Noise-schedule-driven BT-CMDP eta_t."""

    eta_min = float(eta_min)
    eta_max = float(eta_max)
    if hasattr(scheduler, "alphas_cumprod"):
        alphas = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        index = int(timestep.detach().flatten()[0].long().item())
        index = max(0, min(index, int(alphas.shape[0]) - 1))
        alpha_bar = alphas[index]
    else:
        denom = max(1, int(num_steps) - 1)
        alpha_bar = torch.tensor(float(step_idx) / float(denom), device=device, dtype=dtype)
    eta = eta_min + (eta_max - eta_min) * torch.sqrt((1.0 - alpha_bar).clamp(0.0, 1.0))
    return eta.clamp(min(eta_min, eta_max), max(eta_min, eta_max)).view(1, 1, 1, 1)


def image_to_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@dataclass
class ReferenceTrainingMetadata:
    method: str
    paper: str
    base_model: str
    domain: str
    class_name: Optional[str]
    train_samples: int
    resolution: int
    steps: int
    batch_size: int
    learning_rate: float
    crop_dilation: float
    caption: str


class ReferenceDefectCropDataset(Dataset[dict[str, Any]]):
    """BBox defect crops used for reference-paper LDM fine-tuning."""

    def __init__(
        self,
        samples: Sequence[DefectSample],
        *,
        resolution: int,
        domain: str,
        crop_dilation: float = 1.0,
        horizontal_flip: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.resolution = int(resolution)
        self.domain = domain
        self.crop_dilation = max(1.0, float(crop_dilation))
        self.horizontal_flip = bool(horizontal_flip)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        crop_box = expand_bbox(sample.bbox, image.size, self.crop_dilation)
        crop = image.crop(crop_box)
        if self.horizontal_flip and (index % 2 == 1):
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return {
            "pixel_values": image_to_tensor(crop, self.resolution),
            "prompt": reference_caption(self.domain, sample.cls_name),
            "class": sample.cls_name,
            "target_key": sample.target_key,
        }


def _collate_reference_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
    }


class ReferenceLDMTrainer:
    """Full U-Net LDM fine-tuning as described in the reference paper."""

    def __init__(
        self,
        *,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        resolution: int = 512,
        mixed_precision: str = "no",
    ) -> None:
        self.base_model = base_model
        self.resolution = int(resolution)
        self.mixed_precision = mixed_precision

    def _dtype(self) -> torch.dtype:
        if (
            torch.cuda.is_available()
            and self.mixed_precision.lower() in {"bf16", "bfloat16"}
            and torch.cuda.is_bf16_supported()
        ):
            return torch.bfloat16
        if torch.cuda.is_available() and self.mixed_precision.lower() in {"fp16", "float16"}:
            return torch.float16
        return torch.float32

    def train(
        self,
        samples: Sequence[DefectSample],
        output_dir: Path,
        *,
        domain: str,
        class_name: Optional[str] = None,
        steps: int = 1000,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        grad_accum: int = 1,
        warmup_steps: int = 500,
        crop_dilation: float = 1.0,
        max_train_samples: Optional[int] = None,
        seed: int = 42,
    ) -> Path:
        if class_name is not None:
            samples = [sample for sample in samples if sample.cls_name == class_name]
        if max_train_samples is not None:
            samples = list(samples)[: max(1, int(max_train_samples))]
        if not samples:
            raise ValueError("No samples available for reference LDM training.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self._dtype()
        torch.manual_seed(int(seed))

        pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.to(device)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.unet.requires_grad_(True)
        try:
            pipe.unet.enable_gradient_checkpointing()
        except Exception:
            pass

        dataset = ReferenceDefectCropDataset(
            samples,
            resolution=self.resolution,
            domain=domain,
            crop_dilation=crop_dilation,
        )
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=True,
            num_workers=0,
            collate_fn=_collate_reference_batch,
        )
        optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=float(learning_rate))
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=min(int(warmup_steps), int(steps)),
            num_training_steps=int(steps),
        )

        scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        global_step = 0
        grad_accum = max(1, int(grad_accum))
        progress = tqdm(total=int(steps), desc=f"Reference LDM training ({class_name or 'all'})")
        optimizer.zero_grad()
        while global_step < int(steps):
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
                prompts = batch["prompts"]
                input_ids = pipe.tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(device)
                with torch.no_grad():
                    encoder_hidden_states = pipe.text_encoder(input_ids)[0]
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample() * scaling
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                (loss / grad_accum).backward()
                if (global_step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                global_step += 1
                progress.set_postfix({"loss": f"{float(loss.detach().cpu()):.4f}"})
                progress.update(1)
                if global_step >= int(steps):
                    break
        progress.close()

        pipe.save_pretrained(output_dir)
        metadata = ReferenceTrainingMetadata(
            method="reference_ldm_finetune",
            paper="Context-aware defect sample generation using conditional diffusion model for surface defect inspection",
            base_model=self.base_model,
            domain=domain,
            class_name=class_name,
            train_samples=len(samples),
            resolution=self.resolution,
            steps=int(steps),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            crop_dilation=float(crop_dilation),
            caption=reference_caption(domain, class_name) if class_name else "class-aware defect prompts",
        )
        (output_dir / "reference_ldm_config.json").write_text(
            json.dumps(asdict(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            pipe.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output_dir


@dataclass
class CMDPRecord:
    target_key: str
    source_key: str
    class_name: str
    output_path: str
    method: str
    expanded_bbox: tuple[int, int, int, int]
    defect_bbox_in_expanded: tuple[int, int, int, int]
    dilation_factor: float
    background_preservation: float
    boundary_score: float
    total: float
    pseudo_area_ratio: Optional[float] = None
    core_area_ratio: Optional[float] = None
    band_area_ratio: Optional[float] = None
    out_area_ratio: Optional[float] = None
    eta_mean: Optional[float] = None


class ReferenceCMDPGenerator:
    """Reference-paper SCMP + CMDP generator.

    This implementation follows the paper equations directly: after each
    reverse-diffusion step, the defect latent is retained inside the
    original bbox while the expanded surrounding context latent is injected
    outside that bbox.
    """

    def __init__(
        self,
        model_dir: Path | str,
        *,
        scheduler: str = "ddim",
        mixed_precision: str = "fp16",
    ) -> None:
        self.model_dir = str(model_dir)
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        self.pipe: StableDiffusionPipeline | None = None

    def _dtype(self) -> torch.dtype:
        if (
            torch.cuda.is_available()
            and self.mixed_precision.lower() in {"bf16", "bfloat16"}
            and torch.cuda.is_bf16_supported()
        ):
            return torch.bfloat16
        if torch.cuda.is_available() and self.mixed_precision.lower() in {"fp16", "float16"}:
            return torch.float16
        return torch.float32

    def _load_pipe(self) -> StableDiffusionPipeline:
        if self.pipe is not None:
            return self.pipe
        dtype = self._dtype()
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_dir,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if self.scheduler.lower() == "ddpm":
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        self.pipe = pipe
        return pipe

    @staticmethod
    def _encode_prompt(pipe: StableDiffusionPipeline, prompt: str, guidance_scale: float) -> torch.Tensor:
        device = pipe.unet.device
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        text_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            cond = text_encoder(text_inputs.input_ids.to(device))[0]
        if guidance_scale <= 1.0:
            return cond
        uncond_inputs = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond = text_encoder(uncond_inputs.input_ids.to(device))[0]
        return torch.cat([uncond, cond], dim=0)

    @staticmethod
    def _encode_image(pipe: StableDiffusionPipeline, image: Image.Image, resolution: int, dtype: torch.dtype) -> torch.Tensor:
        device = pipe.vae.device
        pixels = image_to_tensor(image, resolution)[None].to(device=device, dtype=dtype)
        scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        with torch.no_grad():
            return pipe.vae.encode(pixels).latent_dist.sample() * scaling

    @staticmethod
    def _decode_latents(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> Image.Image:
        scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        with torch.no_grad():
            image = pipe.vae.decode(latents / scaling).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype(np.uint8)
        return Image.fromarray(image, mode="RGB")

    @staticmethod
    def _add_noise(
        scheduler: Any,
        original: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(scheduler, "add_noise"):
            return scheduler.add_noise(original, noise, timestep)
        alphas = scheduler.alphas_cumprod.to(device=original.device, dtype=original.dtype)
        t = timestep.flatten()[0].long().clamp(0, len(alphas) - 1)
        alpha = alphas[t].view(1, 1, 1, 1)
        return alpha.sqrt() * original + (1.0 - alpha).sqrt() * noise

    @staticmethod
    def _quality(
        original_crop: Image.Image,
        generated_crop: Image.Image,
        defect_mask: Image.Image,
    ) -> tuple[float, float, float]:
        orig = np.asarray(original_crop.convert("RGB"), dtype=np.float32)
        gen = np.asarray(generated_crop.resize(original_crop.size).convert("RGB"), dtype=np.float32)
        mask = np.asarray(defect_mask.resize(original_crop.size).convert("L"), dtype=np.float32) / 255.0
        bg = mask < 0.05
        if bg.any():
            bg_mse = float(np.mean((orig[bg] - gen[bg]) ** 2))
        else:
            bg_mse = 0.0
        bg_score = float(math.exp(-bg_mse / (18.0**2)))

        edge = np.asarray(defect_mask.convert("L"), dtype=np.uint8)
        edge = np.logical_xor(edge > 24, edge > 180)
        if edge.any():
            orig_gray = orig.mean(axis=2)
            gen_gray = gen.mean(axis=2)
            edge_diff = float(np.mean(np.abs(orig_gray[edge] - gen_gray[edge])))
        else:
            edge_diff = 0.0
        boundary_score = float(math.exp(-edge_diff / 28.0))
        total = 0.55 * bg_score + 0.45 * boundary_score
        return bg_score, boundary_score, total

    @staticmethod
    def _quality_domains(
        original_crop: Image.Image,
        generated_crop: Image.Image,
        out_mask: Image.Image,
        band_mask: Image.Image,
    ) -> tuple[float, float, float]:
        orig = np.asarray(original_crop.convert("RGB"), dtype=np.float32)
        gen = np.asarray(generated_crop.resize(original_crop.size).convert("RGB"), dtype=np.float32)
        out = np.asarray(out_mask.resize(original_crop.size).convert("L"), dtype=np.float32) / 255.0
        band = np.asarray(band_mask.resize(original_crop.size).convert("L"), dtype=np.float32) / 255.0
        bg = out > 0.5
        if bg.any():
            bg_mse = float(np.mean((orig[bg] - gen[bg]) ** 2))
        else:
            bg_mse = 0.0
        bg_score = float(math.exp(-bg_mse / (18.0**2)))

        boundary = band > 0.1
        if boundary.any():
            edge_diff = float(np.mean(np.abs(orig.mean(axis=2)[boundary] - gen.mean(axis=2)[boundary])))
        else:
            edge_diff = 0.0
        boundary_score = float(math.exp(-edge_diff / 28.0))
        total = 0.55 * bg_score + 0.45 * boundary_score
        return bg_score, boundary_score, total

    def generate(
        self,
        samples: Sequence[DefectSample],
        output_dir: Path,
        *,
        domain: str,
        dilation_factor: float = 1.3,
        resolution: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        seed: int = 42,
        max_samples: Optional[int] = None,
        masking_mode: str = "reference",
        eta_min: float = 0.30,
        eta_max: float = 0.85,
        dbt_core_radius: int = 2,
        dbt_band_radius: int = 8,
        dbt_small_core_ratio: float = 0.2,
        dbt_out_safety_margin: int = 0,
        dbt_residual_blur_sigma: float = 2.0,
        dbt_gradient_weight: float = 0.5,
        dbt_tau: Optional[float] = None,
        dbt_min_area_ratio: float = 0.01,
        dbt_max_area_ratio: float = 0.25,
        background_propagation_sigma: float = 1.0,
        dbt_mask_geometry: str = "pseudo",
        dbt_box_inner_strength: float = 0.65,
        dbt_box_pseudo_strength: float = 1.0,
        dbt_box_boundary_strength: float = 0.50,
        dbt_stochastic_box: bool = False,
        dbt_stochastic_profile: str = "balanced",
        dbt_shape_preserving: bool = False,
        dbt_core_generation_strength: float = 1.0,
        dbt_core_pixel_strength: float = 1.0,
        dbt_band_pixel_strength: float = 0.55,
        dbt_pixel_residual_clip: Optional[float] = None,
        dbt_luminance_only: bool = False,
        dbt_shape_final_mask: str = "defect",
        dbt_contrast_preservation: float = 0.0,
        dbt_contrast_blur_sigma: float = 3.0,
        dbt_contrast_min: float = 2.5,
    ) -> list[Path]:
        pipe = self._load_pipe()
        device = pipe.unet.device
        dtype = self._dtype()
        output_dir = Path(output_dir)
        normalized_mode = masking_mode.lower().replace("_", "-")
        use_dbt = normalized_mode in {"dbt", "dbt-cmdp", "dbtcmdp"}
        dbt_geometry = str(dbt_mask_geometry).lower().replace("_", "-")
        use_box_geometry = dbt_geometry in {"box", "box-adaptive", "boxadaptive", "bbox", "bbox-adaptive"}
        method = "dbt_cmdp" if use_dbt else "reference_cmdp"
        image_dir = output_dir / "images"
        artifacts_dir = output_dir / "artifacts" / method
        crop_dir = artifacts_dir / "expanded_crops"
        gen_crop_dir = artifacts_dir / "generated_expanded_crops"
        mask_dir = artifacts_dir / "masks"
        for path in (image_dir, crop_dir, gen_crop_dir, mask_dir):
            path.mkdir(parents=True, exist_ok=True)
        scores_path = artifacts_dir / "candidate_scores.jsonl"

        selected_samples = list(samples)
        if max_samples is not None:
            selected_samples = selected_samples[: max(1, int(max_samples))]

        pipe.scheduler.set_timesteps(int(num_inference_steps), device=device)
        do_cfg = guidance_scale > 1.0
        prompt_cache: dict[str, torch.Tensor] = {}
        prompts_by_class: dict[str, str] = {}
        outputs: list[Path] = []
        rows: list[dict[str, Any]] = []

        progress_desc = "DBT-CMDP generation" if use_dbt else "Reference CMDP generation"
        for index, sample in enumerate(tqdm(selected_samples, desc=progress_desc)):
            prompt = reference_caption(domain, sample.cls_name)
            prompts_by_class.setdefault(sample.cls_name, prompt)
            prompt_embeds = prompt_cache.get(prompt)
            if prompt_embeds is None:
                prompt_embeds = self._encode_prompt(pipe, prompt, guidance_scale)
                prompt_cache[prompt] = prompt_embeds
            generator = torch.Generator(device=device).manual_seed(int(seed) + index * 1009)
            original = Image.open(sample.image_path).convert("RGB")
            expanded = expand_bbox(sample.bbox, original.size, dilation_factor)
            local_bbox = bbox_inside_crop(sample.bbox, expanded)
            expanded_crop = original.crop(expanded)
            working_crop = expanded_crop.resize((resolution, resolution), Image.Resampling.LANCZOS)
            scaled_local_bbox = scale_bbox_to_size(local_bbox, expanded_crop.size, working_crop.size)
            dbt_masks: Optional[DBTMaskSet] = None
            mask_stats: dict[str, float | int | bool | str] = {}
            sample_box_params: Optional[StochasticBoxParams] = None
            sample_band_radius = int(dbt_band_radius)
            sample_residual_blur_sigma = float(dbt_residual_blur_sigma)
            sample_gradient_weight = float(dbt_gradient_weight)
            sample_max_area_ratio = float(dbt_max_area_ratio)
            sample_box_inner_strength = float(dbt_box_inner_strength)
            sample_box_pseudo_strength = float(dbt_box_pseudo_strength)
            sample_box_boundary_strength = float(dbt_box_boundary_strength)
            if use_dbt and use_box_geometry and dbt_stochastic_box:
                sample_box_params = sample_stochastic_box_params(
                    sample.cls_name,
                    sample.target_key,
                    index=index,
                    seed=seed,
                    profile=dbt_stochastic_profile,
                    base_band_radius=dbt_band_radius,
                    base_residual_blur_sigma=dbt_residual_blur_sigma,
                    base_gradient_weight=dbt_gradient_weight,
                    base_max_area_ratio=dbt_max_area_ratio,
                    base_box_inner_strength=dbt_box_inner_strength,
                    base_box_pseudo_strength=dbt_box_pseudo_strength,
                    base_box_boundary_strength=dbt_box_boundary_strength,
                )
                sample_band_radius = sample_box_params.band_radius
                sample_residual_blur_sigma = sample_box_params.residual_blur_sigma
                sample_gradient_weight = sample_box_params.gradient_weight
                sample_max_area_ratio = sample_box_params.max_area_ratio
                sample_box_inner_strength = sample_box_params.box_inner_strength
                sample_box_pseudo_strength = sample_box_params.box_pseudo_strength
                sample_box_boundary_strength = sample_box_params.box_boundary_strength
            if use_dbt:
                if use_box_geometry:
                    dbt_masks = build_box_adaptive_pixel_masks(
                        working_crop,
                        scaled_local_bbox,
                        residual_blur_sigma=sample_residual_blur_sigma,
                        gradient_weight=sample_gradient_weight,
                        tau=dbt_tau,
                        band_radius=sample_band_radius,
                        min_area_ratio=dbt_min_area_ratio,
                        max_area_ratio=sample_max_area_ratio,
                    )
                else:
                    dbt_masks = build_dbt_pixel_masks(
                        working_crop,
                        scaled_local_bbox,
                        residual_blur_sigma=dbt_residual_blur_sigma,
                        gradient_weight=dbt_gradient_weight,
                        tau=dbt_tau,
                        core_radius=dbt_core_radius,
                        band_radius=sample_band_radius,
                        small_core_ratio=dbt_small_core_ratio,
                        out_safety_margin=dbt_out_safety_margin,
                        min_area_ratio=dbt_min_area_ratio,
                        max_area_ratio=dbt_max_area_ratio,
                    )
                context_mask = dbt_masks.out
                defect_mask = dbt_masks.defect
                mask_stats = dbt_masks.stats
            else:
                context_mask, defect_mask = build_scmp_pixel_masks(working_crop.size, scaled_local_bbox)
            crop_dir.joinpath(f"{sample.target_key}.png").parent.mkdir(parents=True, exist_ok=True)
            working_crop.save(crop_dir / f"{sample.target_key}.png")
            if use_dbt and dbt_masks is not None:
                dbt_masks.pseudo.save(mask_dir / f"{sample.target_key}_pseudo_P.png")
                dbt_masks.core.save(mask_dir / f"{sample.target_key}_M_core.png")
                dbt_masks.band.save(mask_dir / f"{sample.target_key}_M_band.png")
                dbt_masks.out.save(mask_dir / f"{sample.target_key}_M_out.png")
                dbt_masks.defect.save(mask_dir / f"{sample.target_key}_M_core_plus_band.png")
            else:
                context_mask.save(mask_dir / f"{sample.target_key}_context_M.png")
                defect_mask.save(mask_dir / f"{sample.target_key}_defect_1minusM.png")

            context_latents = self._encode_image(pipe, working_crop, resolution, dtype)
            latent_h, latent_w = context_latents.shape[-2:]
            if use_dbt and dbt_masks is not None:
                latent_core_mask = mask_to_latent_tensor(dbt_masks.core, (latent_h, latent_w), device, context_latents.dtype)
                latent_band_mask = mask_to_latent_tensor(dbt_masks.band, (latent_h, latent_w), device, context_latents.dtype)
                latent_out_mask = mask_to_latent_tensor(dbt_masks.out, (latent_h, latent_w), device, context_latents.dtype)
                latent_pseudo_prior_mask = mask_to_latent_tensor(
                    dbt_masks.pseudo,
                    (latent_h, latent_w),
                    device,
                    context_latents.dtype,
                )
                latent_core_mask, latent_band_mask, latent_out_mask = normalize_latent_triplet(
                    latent_core_mask,
                    latent_band_mask,
                    latent_out_mask,
                )
                latent_pseudo_prior_mask = latent_pseudo_prior_mask.clamp(0.0, 1.0)
                latent_context_mask = latent_out_mask
                latent_defect_mask = latent_core_mask + latent_band_mask
            else:
                latent_context_mask = mask_to_latent_tensor(context_mask, (latent_h, latent_w), device, context_latents.dtype)
                latent_defect_mask = 1.0 - latent_context_mask
                latent_core_mask = latent_defect_mask
                latent_band_mask = torch.zeros_like(latent_defect_mask)
                latent_out_mask = latent_context_mask
                latent_pseudo_prior_mask = torch.zeros_like(latent_defect_mask)
            context_noise = torch.randn(
                context_latents.shape,
                device=device,
                dtype=context_latents.dtype,
                generator=generator,
            )
            latents = torch.randn(
                context_latents.shape,
                device=device,
                dtype=context_latents.dtype,
                generator=generator,
            )

            timesteps = pipe.scheduler.timesteps
            eta_values: list[float] = []
            with torch.no_grad():
                for step_idx, timestep in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
                    noise_pred = pipe.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample
                    if do_cfg:
                        noise_uncond, noise_text = noise_pred.chunk(2)
                        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                    refined = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
                    if step_idx + 1 < len(timesteps):
                        next_t = timesteps[step_idx + 1].reshape(1).to(device=device)
                        context_for_next = self._add_noise(pipe.scheduler, context_latents, context_noise, next_t)
                    else:
                        context_for_next = context_latents
                    if use_dbt:
                        eta = transition_eta(
                            pipe.scheduler,
                            timestep.reshape(1).to(device=device),
                            eta_min=eta_min,
                            eta_max=eta_max,
                            step_idx=step_idx,
                            num_steps=len(timesteps),
                            device=device,
                            dtype=context_latents.dtype,
                        )
                        propagated_background = propagate_background_latents(
                            context_for_next,
                            latent_out_mask,
                            sigma=background_propagation_sigma,
                        )
                        if use_box_geometry:
                            boundary_strength = min(1.0, max(0.0, float(sample_box_boundary_strength)))
                            band_eta = (eta * boundary_strength).clamp(0.0, 1.0)
                        else:
                            band_eta = eta
                        transition_latents = (1.0 - band_eta) * propagated_background + band_eta * refined
                        core_strength = min(1.0, max(0.0, float(dbt_core_generation_strength)))
                        if use_box_geometry:
                            inner_strength = min(1.0, max(0.0, float(sample_box_inner_strength)))
                            pseudo_strength = min(1.0, max(inner_strength, float(sample_box_pseudo_strength)))
                            core_strength = inner_strength + (pseudo_strength - inner_strength) * latent_pseudo_prior_mask
                        core_latents = (1.0 - core_strength) * context_for_next + core_strength * refined
                        latents = (
                            latent_out_mask * context_for_next
                            + latent_core_mask * core_latents
                            + latent_band_mask * transition_latents
                        )
                        eta_values.append(float(eta.detach().cpu().flatten()[0]))
                    else:
                        latents = refined * latent_defect_mask + context_for_next * latent_context_mask

            generated_crop = self._decode_latents(pipe, latents).resize(expanded_crop.size, Image.Resampling.LANCZOS)
            if use_dbt and dbt_masks is not None and dbt_shape_preserving:
                generated_crop = shape_preserving_pixel_blend(
                    expanded_crop,
                    generated_crop,
                    dbt_masks.core,
                    dbt_masks.band,
                    core_strength=dbt_core_pixel_strength,
                    band_strength=dbt_band_pixel_strength,
                    residual_clip=dbt_pixel_residual_clip,
                    luminance_only=dbt_luminance_only,
                    contrast_preservation=dbt_contrast_preservation,
                    contrast_blur_sigma=dbt_contrast_blur_sigma,
                    contrast_min=dbt_contrast_min,
                )
            gen_crop_dir.joinpath(f"{sample.target_key}.png").parent.mkdir(parents=True, exist_ok=True)
            generated_crop.save(gen_crop_dir / f"{sample.target_key}.png")

            expanded_canvas = original.copy()
            expanded_canvas.paste(generated_crop, expanded[:2])
            full_mask = Image.new("L", original.size, 0)
            if use_dbt:
                final_mask_mode = str(dbt_shape_final_mask).lower().replace("_", "-")
                if dbt_shape_preserving and dbt_masks is not None and final_mask_mode in {"pseudo", "p"}:
                    crop_mask = dbt_masks.pseudo.resize(expanded_crop.size, Image.Resampling.BILINEAR)
                else:
                    crop_mask = defect_mask.resize(expanded_crop.size, Image.Resampling.BILINEAR)
                full_mask.paste(crop_mask, expanded[:2])
            else:
                ImageDraw.Draw(full_mask).rectangle(sample.bbox, fill=255)
            full_mask = full_mask.filter(ImageFilter.GaussianBlur(radius=2.0))
            final = create_composite_preserving_background(original, expanded_canvas, full_mask)

            out_path = image_dir / f"{sample.target_key}.png"
            final.save(out_path)
            generated_working_crop = generated_crop.resize(working_crop.size, Image.Resampling.LANCZOS)
            if use_dbt and dbt_masks is not None:
                bg_score, boundary_score, total = self._quality_domains(
                    working_crop,
                    generated_working_crop,
                    dbt_masks.out,
                    dbt_masks.band,
                )
            else:
                bg_score, boundary_score, total = self._quality(working_crop, generated_working_crop, defect_mask)
            record = CMDPRecord(
                target_key=sample.target_key,
                source_key=sample.source_stem,
                class_name=sample.cls_name,
                output_path=str(out_path.resolve()),
                method=method,
                expanded_bbox=expanded,
                defect_bbox_in_expanded=local_bbox,
                dilation_factor=float(dilation_factor),
                background_preservation=bg_score,
                boundary_score=boundary_score,
                total=total,
                pseudo_area_ratio=(
                    float(mask_stats["pseudo_area_ratio"]) if "pseudo_area_ratio" in mask_stats else None
                ),
                core_area_ratio=float(mask_stats["core_area_ratio"]) if "core_area_ratio" in mask_stats else None,
                band_area_ratio=float(mask_stats["band_area_ratio"]) if "band_area_ratio" in mask_stats else None,
                out_area_ratio=float(mask_stats["out_area_ratio"]) if "out_area_ratio" in mask_stats else None,
                eta_mean=float(np.mean(eta_values)) if eta_values else None,
            )
            rows.append(
                {
                    "target_key": record.target_key,
                    "source_key": record.source_key,
                    "class": record.class_name,
                    "prompt": prompt,
                    "selected": True,
                    "candidate_path": str(out_path.resolve()),
                    "method": method,
                    "paper_reference": (
                        "DBT-CMDP over Measurement 2026 SCMP + CMDP"
                        if use_dbt
                        else "Measurement 2026, SCMP + CMDP"
                    ),
                    "quality": {
                        "background_preservation": record.background_preservation,
                        "boundary_score": record.boundary_score,
                        "total": record.total,
                        "outside_delta": 1.0 - record.background_preservation,
                    },
                    "cmdp": asdict(record),
                    "dbt": {
                        "mask_stats": mask_stats,
                        "eta_min": float(eta_min),
                        "eta_max": float(eta_max),
                        "eta_mean": record.eta_mean,
                        "background_propagation_sigma": float(background_propagation_sigma),
                        "mask_geometry": dbt_geometry,
                        "band_radius": int(sample_band_radius),
                        "residual_blur_sigma": float(sample_residual_blur_sigma),
                        "gradient_weight": float(sample_gradient_weight),
                        "max_area_ratio": float(sample_max_area_ratio),
                        "box_inner_strength": float(sample_box_inner_strength),
                        "box_pseudo_strength": float(sample_box_pseudo_strength),
                        "box_boundary_strength": float(sample_box_boundary_strength),
                        "stochastic_box": bool(dbt_stochastic_box and use_box_geometry),
                        "stochastic_profile": str(dbt_stochastic_profile),
                        "stochastic_family": sample_box_params.family if sample_box_params is not None else None,
                        "shape_preserving": bool(dbt_shape_preserving),
                        "core_generation_strength": float(dbt_core_generation_strength),
                        "core_pixel_strength": float(dbt_core_pixel_strength),
                        "band_pixel_strength": float(dbt_band_pixel_strength),
                        "pixel_residual_clip": (
                            float(dbt_pixel_residual_clip) if dbt_pixel_residual_clip is not None else None
                        ),
                        "luminance_only": bool(dbt_luminance_only),
                        "shape_final_mask": str(dbt_shape_final_mask),
                        "contrast_preservation": float(dbt_contrast_preservation),
                        "contrast_blur_sigma": float(dbt_contrast_blur_sigma),
                        "contrast_min": float(dbt_contrast_min),
                    }
                    if use_dbt
                    else None,
                }
            )
            outputs.append(out_path)

        score_mode = "a" if scores_path.exists() else "w"
        with scores_path.open(score_mode, encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        run_context = {
            "method": method,
            "paper": "Context-aware defect sample generation using conditional diffusion model for surface defect inspection",
            "model_dir": self.model_dir,
            "domain": domain,
            "prompt_mode": "class-aware",
            "prompts_by_class": dict(sorted(prompts_by_class.items())),
            "dilation_factor": float(dilation_factor),
            "resolution": int(resolution),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
            "generated_images": len(outputs),
            "score_path": str(scores_path.resolve()),
        }
        if use_dbt:
            run_context["dbt"] = {
                "eta_min": float(eta_min),
                "eta_max": float(eta_max),
                "core_radius": int(dbt_core_radius),
                "band_radius": int(dbt_band_radius),
                "small_core_ratio": float(dbt_small_core_ratio),
                "out_safety_margin": int(dbt_out_safety_margin),
                "residual_blur_sigma": float(dbt_residual_blur_sigma),
                "gradient_weight": float(dbt_gradient_weight),
                "tau": float(dbt_tau) if dbt_tau is not None else None,
                "min_area_ratio": float(dbt_min_area_ratio),
                "max_area_ratio": float(dbt_max_area_ratio),
                "background_propagation_sigma": float(background_propagation_sigma),
                "mask_geometry": dbt_geometry,
                "box_inner_strength": float(dbt_box_inner_strength),
                "box_pseudo_strength": float(dbt_box_pseudo_strength),
                "box_boundary_strength": float(dbt_box_boundary_strength),
                "stochastic_box": bool(dbt_stochastic_box and use_box_geometry),
                "stochastic_profile": str(dbt_stochastic_profile),
                "shape_preserving": bool(dbt_shape_preserving),
                "core_generation_strength": float(dbt_core_generation_strength),
                "core_pixel_strength": float(dbt_core_pixel_strength),
                "band_pixel_strength": float(dbt_band_pixel_strength),
                "pixel_residual_clip": float(dbt_pixel_residual_clip) if dbt_pixel_residual_clip is not None else None,
                "luminance_only": bool(dbt_luminance_only),
                "shape_final_mask": str(dbt_shape_final_mask),
                "contrast_preservation": float(dbt_contrast_preservation),
                "contrast_blur_sigma": float(dbt_contrast_blur_sigma),
                "contrast_min": float(dbt_contrast_min),
            }
        (output_dir / "run_context.json").write_text(
            json.dumps(run_context, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return outputs

    def release(self) -> None:
        if self.pipe is not None:
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DBTCMDPGenerator(ReferenceCMDPGenerator):
    """DBT-CMDP generator wrapper for the proposed tri-domain CMDP variant."""

    def generate(self, samples: Sequence[DefectSample], output_dir: Path, **kwargs: Any) -> list[Path]:
        kwargs["masking_mode"] = "dbt"
        return super().generate(samples, output_dir, **kwargs)
