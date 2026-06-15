from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..data.loader import DefectSample


@dataclass(frozen=True)
class MorphologyProfile:
    key: str
    cls_name: str
    bbox: tuple[int, int, int, int]
    bbox_aspect: float
    bbox_area_ratio: float
    weak_coverage: float
    component_count: int
    elongation: float
    orientation_deg: float
    boundary_roughness: float


@dataclass
class MorphologyMaskPlan:
    target_key: str
    cls_name: str
    profile_key: str
    target_bbox: tuple[int, int, int, int]
    scaled_bbox: tuple[int, int, int, int]
    candidate_index: int
    seed: int
    family: str
    target_coverage: float
    actual_coverage: float
    component_count: int
    orientation_deg: float
    mask_source: str


class MorphologyPrior:
    """Class-wise weak morphology statistics mined from existing defect boxes."""

    def __init__(self, profiles_by_class: dict[str, list[MorphologyProfile]]):
        self.profiles_by_class = {cls: list(rows) for cls, rows in profiles_by_class.items()}

    @classmethod
    def from_samples(cls, samples: Sequence[DefectSample]) -> "MorphologyPrior":
        profiles_by_class: dict[str, list[MorphologyProfile]] = {}
        for sample in samples:
            try:
                image = _read_rgb(sample.image_path)
                profile = _profile_from_sample(sample, image)
            except Exception:
                profile = _fallback_profile(sample)
            profiles_by_class.setdefault(sample.cls_name, []).append(profile)
        return cls(profiles_by_class)

    def sample_profile(self, cls_name: str, *, seed: int = 0) -> MorphologyProfile:
        profiles = self.profiles_by_class.get(cls_name) or []
        if not profiles:
            return MorphologyProfile(
                key=f"{cls_name}_fallback",
                cls_name=cls_name,
                bbox=(0, 0, 32, 32),
                bbox_aspect=1.0,
                bbox_area_ratio=0.04,
                weak_coverage=0.22,
                component_count=1,
                elongation=1.0,
                orientation_deg=0.0,
                boundary_roughness=1.15,
            )
        return random.Random(seed).choice(profiles)

    def write_manifest(self, path: Path) -> None:
        rows: list[dict[str, object]] = []
        for cls_name, profiles in sorted(self.profiles_by_class.items()):
            coverages = [p.weak_coverage for p in profiles]
            components = [p.component_count for p in profiles]
            rows.append(
                {
                    "class": cls_name,
                    "count": len(profiles),
                    "coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
                    "coverage_median": float(np.median(coverages)) if coverages else 0.0,
                    "component_median": float(np.median(components)) if components else 0.0,
                    "profiles": [asdict(p) for p in profiles],
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def dilate_binary_mask(mask: np.ndarray, iterations: int = 1, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((max(1, int(kernel_size)), max(1, int(kernel_size))), dtype=np.uint8)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    return cv2.dilate(binary, kernel, iterations=max(1, int(iterations))).astype(bool)


def scale_bbox(
    bbox: tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_w, src_h = source_size
    dst_w, dst_h = target_size
    xmin, ymin, xmax, ymax = bbox
    return _clip_bbox(
        (
            int(xmin * dst_w / max(src_w, 1)),
            int(ymin * dst_h / max(src_h, 1)),
            int(xmax * dst_w / max(src_w, 1)),
            int(ymax * dst_h / max(src_h, 1)),
        ),
        dst_w,
        dst_h,
    )


def build_morphology_calibrated_mask(
    prior: MorphologyPrior,
    cls_name: str,
    target_bbox: tuple[int, int, int, int],
    *,
    target_key: str = "",
    target_size: tuple[int, int] = (512, 512),
    target_source_size: tuple[int, int] = (200, 200),
    candidate_index: int = 0,
    seed: int = 0,
    feather_radius: float = 2.4,
) -> tuple[Image.Image, MorphologyMaskPlan]:
    rng = random.Random(seed)
    profile = prior.sample_profile(cls_name, seed=seed)
    scaled = scale_bbox(target_bbox, target_source_size, target_size)
    x0, y0, x1, y1 = scaled
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    family = _family_from_class(cls_name)

    coverage = profile.weak_coverage * rng.uniform(0.78, 1.18)
    if family == "scratch":
        coverage = _clamp(coverage, 0.010, 0.12)
    elif family == "crack":
        coverage = _clamp(coverage, 0.018, 0.20)
    elif family == "pitted":
        coverage = _clamp(coverage, 0.014, 0.13)
    elif family == "inclusion":
        coverage = _clamp(coverage, 0.020, 0.095)
    elif family == "patch":
        coverage = _clamp(max(coverage, 0.12), 0.10, 0.38)
    elif family == "scale":
        coverage = _clamp(coverage, 0.08, 0.42)
    else:
        coverage = _clamp(coverage, 0.06, 0.48)

    component_count = max(1, int(round(profile.component_count * rng.uniform(0.75, 1.35))))
    orientation = profile.orientation_deg + rng.uniform(-18.0, 18.0)

    if family == "scratch":
        full = _draw_scratch_family(target_size, scaled, coverage, component_count, orientation, rng)
    elif family == "crack":
        full = _draw_crack_family(target_size, scaled, coverage, component_count, orientation, rng)
    elif family == "pitted":
        full = _draw_pitted_family(target_size, scaled, coverage, component_count, rng)
    elif family == "scale":
        full = _draw_scale_family(target_size, scaled, coverage, component_count, orientation, rng)
    elif family == "patch":
        full = _draw_scale_family(target_size, scaled, coverage, max(component_count, 2), orientation, rng)
    elif family == "inclusion":
        full = _draw_inclusion_family(target_size, scaled, coverage, component_count, orientation, rng)
    else:
        full = _draw_blob_family(target_size, scaled, coverage, component_count, rng)

    full = _constrain_to_bbox(full, scaled)
    actual_coverage = float(full[y0:y1, x0:x1].mean()) if bbox_w * bbox_h else 0.0
    source = "morphology_prior"
    if actual_coverage < 0.006:
        full = _fallback_ellipse_mask(target_size, scaled)
        actual_coverage = float(full[y0:y1, x0:x1].mean()) if bbox_w * bbox_h else 1.0
        source = "ellipse_fallback"

    mask = Image.fromarray((full.astype(np.uint8) * 255)).convert("L")
    feather = _effective_feather_radius(family, feather_radius)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    plan = MorphologyMaskPlan(
        target_key=target_key,
        cls_name=cls_name,
        profile_key=profile.key,
        target_bbox=target_bbox,
        scaled_bbox=scaled,
        candidate_index=candidate_index,
        seed=seed,
        family=family,
        target_coverage=float(coverage),
        actual_coverage=float(actual_coverage),
        component_count=int(component_count),
        orientation_deg=float(orientation),
        mask_source=source,
    )
    return mask, plan


def _read_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    xmin = max(0, min(width - 1, xmin))
    xmax = max(xmin + 1, min(width, xmax))
    ymin = max(0, min(height - 1, ymin))
    ymax = max(ymin + 1, min(height, ymax))
    return xmin, ymin, xmax, ymax


def _weak_mask_from_bbox_crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(bbox, width, height)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return np.ones((max(1, y1 - y0), max(1, x1 - x0)), dtype=bool)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    residual = cv2.absdiff(gray, smooth)
    if int(residual.max()) <= 2:
        contrast = cv2.absdiff(gray, np.full_like(gray, int(np.median(gray))))
        residual = contrast
    _, otsu = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = otsu > 0

    area = float(mask.mean()) if mask.size else 0.0
    if area < 0.015 or area > 0.88:
        threshold = np.percentile(residual, 78)
        mask = residual >= threshold
    if float(mask.mean()) < 0.01:
        h, w = gray.shape
        fallback = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(fallback, (w // 2, h // 2), (max(2, w // 4), max(2, h // 4)), 0, 0, 360, 1, -1)
        mask = fallback > 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def _profile_from_sample(sample: DefectSample, image: np.ndarray) -> MorphologyProfile:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(sample.bbox, width, height)
    weak = _weak_mask_from_bbox_crop(image, (x0, y0, x1, y1))
    coverage = float(weak.mean()) if weak.size else 0.0
    area_ratio = float(((x1 - x0) * (y1 - y0)) / max(1, width * height))
    aspect = float(max(x1 - x0, y1 - y0) / max(1, min(x1 - x0, y1 - y0)))
    components = _component_count(weak)
    elongation, orientation, roughness = _shape_descriptors(weak)
    return MorphologyProfile(
        key=sample.image_path.stem,
        cls_name=sample.cls_name,
        bbox=(x0, y0, x1, y1),
        bbox_aspect=aspect,
        bbox_area_ratio=area_ratio,
        weak_coverage=float(_clamp(coverage, 0.01, 0.85)),
        component_count=components,
        elongation=elongation,
        orientation_deg=orientation,
        boundary_roughness=roughness,
    )


def _fallback_profile(sample: DefectSample) -> MorphologyProfile:
    x0, y0, x1, y1 = sample.bbox
    aspect = max(x1 - x0, y1 - y0) / max(1, min(x1 - x0, y1 - y0))
    family = _family_from_class(sample.cls_name)
    coverage = 0.10 if family == "crack" else 0.22
    components = 4 if family in {"crack", "pitted"} else 1
    return MorphologyProfile(
        key=sample.image_path.stem,
        cls_name=sample.cls_name,
        bbox=sample.bbox,
        bbox_aspect=float(aspect),
        bbox_area_ratio=0.04,
        weak_coverage=coverage,
        component_count=components,
        elongation=float(aspect),
        orientation_deg=0.0,
        boundary_roughness=1.2,
    )


def _shape_descriptors(mask: np.ndarray) -> tuple[float, float, float]:
    binary = mask.astype(np.uint8)
    contours, _hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0, 0.0, 1.0
    contour = max(contours, key=cv2.contourArea)
    area = max(1.0, float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    rect = cv2.minAreaRect(contour)
    (rw, rh) = rect[1]
    short = max(1.0, min(float(rw), float(rh)))
    long = max(short, max(float(rw), float(rh)))
    roughness = perimeter / max(1.0, 2.0 * math.sqrt(math.pi * area))
    return float(long / short), float(rect[2]), float(_clamp(roughness, 0.6, 8.0))


def _component_count(mask: np.ndarray) -> int:
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = max(3, int(mask.size * 0.003))
    count = 0
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            count += 1
    return max(1, count)


def _family_from_class(cls_name: str) -> str:
    cls = cls_name.lower()
    if "scratch" in cls:
        return "scratch"
    if any(key in cls for key in ("crazing", "crack")):
        return "crack"
    if "pit" in cls:
        return "pitted"
    if "inclusion" in cls:
        return "inclusion"
    if "patch" in cls:
        return "patch"
    if "scale" in cls or "rolled" in cls:
        return "scale"
    return "blob"


def _effective_feather_radius(family: str, requested: float) -> float:
    if requested <= 0:
        return requested
    if family == "scratch":
        return min(requested, 0.85)
    if family == "pitted":
        return min(requested, 0.75)
    if family == "inclusion":
        return min(requested, 1.65)
    return requested


def _draw_crack_family(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    coverage: float,
    component_count: int,
    orientation: float,
    rng: random.Random,
) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    branches = max(1, min(7, component_count + rng.randint(0, 2)))
    target_area = max(3.0, coverage * bw * bh)
    length = max(4.0, math.hypot(bw, bh) * 0.75)
    thickness = max(1, min(8, int(round(target_area / max(1.0, branches * length)))))
    base_angle = math.radians(orientation)

    for branch in range(branches):
        angle = base_angle + rng.uniform(-0.55, 0.55)
        cx = rng.uniform(x0 + bw * 0.20, x1 - bw * 0.20)
        cy = rng.uniform(y0 + bh * 0.20, y1 - bh * 0.20)
        steps = rng.randint(4, 8)
        step = length / max(1, steps)
        px = cx - math.cos(angle) * length * 0.35
        py = cy - math.sin(angle) * length * 0.35
        points = []
        local_angle = angle
        for _ in range(steps + 1):
            points.append((_clamp(px, x0, x1), _clamp(py, y0, y1)))
            local_angle += rng.gauss(0.0, 0.18)
            px += math.cos(local_angle) * step
            py += math.sin(local_angle) * step
        draw.line(points, fill=255, width=max(1, thickness + (branch % 2)), joint="curve")

    arr = np.asarray(mask, dtype=np.uint8) > 0
    if rng.random() < 0.5:
        arr = dilate_binary_mask(arr, 1)
    return arr


def _draw_scratch_family(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    coverage: float,
    component_count: int,
    orientation: float,
    rng: random.Random,
) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    strokes = max(1, min(3, component_count + rng.randint(0, 1)))
    target_area = max(2.0, coverage * bw * bh)
    length = max(7.0, math.hypot(bw, bh) * rng.uniform(0.82, 1.10))
    thickness = max(1, min(4, int(round(target_area / max(1.0, strokes * length)))))

    # Scratches in NEU-DET are usually long, bright, and fairly straight.
    if bh >= bw * 1.15:
        base_angle = math.pi / 2 + rng.uniform(-0.28, 0.28)
    else:
        base_angle = math.radians(orientation) + rng.uniform(-0.18, 0.18)

    for _ in range(strokes):
        angle = base_angle + rng.uniform(-0.18, 0.18)
        cx = rng.uniform(x0 + bw * 0.18, x1 - bw * 0.18)
        cy = rng.uniform(y0 + bh * 0.18, y1 - bh * 0.18)
        steps = rng.randint(3, 6)
        step = length / max(1, steps)
        px = cx - math.cos(angle) * length * 0.45
        py = cy - math.sin(angle) * length * 0.45
        points = []
        local_angle = angle
        for _idx in range(steps + 1):
            points.append((_clamp(px, x0, x1), _clamp(py, y0, y1)))
            local_angle += rng.gauss(0.0, 0.045)
            px += math.cos(local_angle) * step
            py += math.sin(local_angle) * step
        draw.line(points, fill=255, width=thickness, joint="curve")

    arr = np.asarray(mask, dtype=np.uint8) > 0
    if thickness <= 1 and rng.random() < 0.45:
        arr = dilate_binary_mask(arr, 1)
    return arr


def _draw_inclusion_family(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    coverage: float,
    component_count: int,
    orientation: float,
    rng: random.Random,
) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    crop = Image.new("L", (bw, bh), 0)
    draw = ImageDraw.Draw(crop)
    streaks = max(1, min(2, component_count + rng.randint(0, 1)))
    target_area = max(6.0, coverage * bw * bh / streaks)
    base_len = _clamp(math.sqrt(target_area) * rng.uniform(3.4, 5.2), max(6.0, bh * 0.34), bh * 0.92)
    base_width = _clamp(target_area / max(2.0, 1.65 * base_len), 0.9, max(1.4, bw * 0.075))

    for _ in range(streaks):
        cx = rng.uniform(bw * 0.25, bw * 0.75)
        cy = rng.uniform(bh * 0.20, bh * 0.80)
        length = base_len * rng.uniform(0.7, 1.25)
        segments = rng.randint(6, 11)
        points = []
        for idx in range(segments):
            t = idx / max(1, segments - 1)
            yy = cy - length / 2 + t * length
            center_jitter = rng.gauss(0.0, max(0.6, bw * 0.018))
            points.append((cx + center_jitter, yy))
        width_px = max(1, int(round(base_width * rng.uniform(1.1, 2.2))))
        draw.line(points, fill=255, width=width_px, joint="curve")
        for px, py in points[1:-1:2]:
            r = base_width * rng.uniform(0.7, 1.5)
            draw.ellipse([px - r, py - r * 1.25, px + r, py + r * 1.25], fill=255)

    if bh >= bw * 1.20:
        angle = rng.uniform(-7.0, 7.0)
    else:
        angle = _clamp(orientation, -18.0, 18.0) if abs(orientation) > 8 else rng.uniform(-12.0, 12.0)
    crop = crop.filter(ImageFilter.GaussianBlur(radius=0.65)).rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
    full = Image.new("L", size, 0)
    full.paste(crop, (x0, y0))
    arr = np.asarray(full, dtype=np.uint8) > 36
    arr = cv2.morphologyEx(arr.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)).astype(bool)
    return arr


def _draw_blob_family(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    coverage: float,
    component_count: int,
    rng: random.Random,
) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    blobs = max(1, min(5, component_count))
    target_area = max(8.0, coverage * bw * bh / blobs)
    radius = math.sqrt(target_area / math.pi)

    for _ in range(blobs):
        cx = rng.uniform(x0 + bw * 0.22, x1 - bw * 0.22)
        cy = rng.uniform(y0 + bh * 0.22, y1 - bh * 0.22)
        vertices = rng.randint(9, 16)
        rx = _clamp(radius * rng.uniform(0.9, 1.7), 2.0, bw * 0.48)
        ry = _clamp(radius * rng.uniform(0.65, 1.35), 2.0, bh * 0.48)
        points = []
        phase = rng.uniform(0, 2 * math.pi)
        for idx in range(vertices):
            angle = phase + 2.0 * math.pi * idx / vertices
            jitter = rng.uniform(0.72, 1.28)
            points.append((cx + math.cos(angle) * rx * jitter, cy + math.sin(angle) * ry * jitter))
        draw.polygon(points, fill=255)

    arr = np.asarray(mask, dtype=np.uint8) > 0
    arr = cv2.morphologyEx(arr.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    return arr


def _draw_pitted_family(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    coverage: float,
    component_count: int,
    rng: random.Random,
) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    crop = Image.new("L", (bw, bh), 0)
    draw = ImageDraw.Draw(crop)
    pits = max(8, min(34, component_count * 3 + rng.randint(3, 10)))
    target_area = max(3.0, coverage * bw * bh / pits)
    radius = _clamp(math.sqrt(target_area / math.pi) * 0.64, 0.75, max(1.6, min(bw, bh) * 0.055))
    for _ in range(pits):
        cx = rng.uniform(bw * 0.08, bw * 0.92)
        cy = rng.uniform(bh * 0.08, bh * 0.92)
        r = radius * rng.uniform(0.65, 1.45)
        vertices = rng.randint(7, 12)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        points = []
        for idx in range(vertices):
            angle = phase + 2.0 * math.pi * idx / vertices
            jitter = rng.uniform(0.58, 1.38)
            rx = r * rng.uniform(0.75, 1.30)
            ry = r * rng.uniform(0.65, 1.25)
            points.append((cx + math.cos(angle) * rx * jitter, cy + math.sin(angle) * ry * jitter))
        draw.polygon(points, fill=255)

    arr = np.asarray(crop, dtype=np.uint8)
    np_rng = np.random.default_rng(rng.randrange(0, 2**32))
    grain = np_rng.normal(0.0, 1.0, arr.shape).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.55)
    arr = np.where(grain > 1.05, 0, arr)
    arr = cv2.morphologyEx((arr > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    full = np.zeros((height, width), dtype=bool)
    full[y0:y1, x0:x1] = arr.astype(bool)
    return full


def _draw_scale_family(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    coverage: float,
    component_count: int,
    orientation: float,
    rng: random.Random,
) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    crop = Image.new("L", (bw, bh), 0)
    draw = ImageDraw.Draw(crop)
    scales = max(2, min(8, component_count + rng.randint(1, 3)))
    target_area = max(6.0, coverage * bw * bh / scales)
    rx = _clamp(math.sqrt(target_area) * rng.uniform(1.1, 1.9), 2.0, bw * 0.38)
    ry = _clamp(math.sqrt(target_area) * rng.uniform(0.35, 0.75), 1.2, bh * 0.25)
    for _ in range(scales):
        cx = rng.uniform(bw * 0.08, bw * 0.92)
        cy = rng.uniform(bh * 0.12, bh * 0.88)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    crop = crop.rotate(orientation, resample=Image.Resampling.BILINEAR, expand=False)
    full = Image.new("L", size, 0)
    full.paste(crop, (x0, y0))
    return np.asarray(full, dtype=np.uint8) > 0


def _fallback_ellipse_mask(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = _clip_bbox(bbox, width, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    axes = (max(2, (x1 - x0) // 3), max(2, (y1 - y0) // 3))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask.astype(bool)


def _constrain_to_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = mask.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(bbox, width, height)
    constrained = np.zeros((height, width), dtype=bool)
    constrained[y0:y1, x0:x1] = mask[y0:y1, x0:x1].astype(bool)
    return constrained


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))

