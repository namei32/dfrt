from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover - optional acceleration only
    cv2 = None


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "gc10": ROOT / "data" / "GC10" / "formal_lora_native" / "gc10_drft_lora_native_ratio_100_dataset",
    "mt": ROOT / "data" / "MT" / "formal_lora_native" / "mt_drft_lora_native_ratio_100_dataset",
    "tilda": ROOT / "data" / "TILDA" / "formal_lora_native" / "tilda_drft_lora_native_ratio_100_dataset",
}

BLUE = (36, 101, 224)
RED = (218, 56, 42)
TEXT = (34, 34, 34)
MUTED = (95, 95, 95)
BG = (248, 248, 246)
PANEL = (255, 255, 255)


@dataclass
class Candidate:
    dataset: str
    class_id: int
    class_name: str
    original_image: str
    generated_image: str
    label: str
    bbox_yolo: tuple[float, float, float, float]
    bbox_area_pct: float
    mean_abs_diff: float
    p95_abs_diff: float
    changed_pct_gt8: float
    changed_pct_gt16: float
    edge_jaccard_distance: float
    largest_component_pct: float
    structure_score: float


def load_names(data_yaml: Path) -> dict[int, str]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def read_boxes(label_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        boxes.append((int(float(parts[0])), tuple(float(v) for v in parts[1:5])))  # type: ignore[arg-type]
    return boxes


def yolo_to_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = box
    x1 = int(max(0, round((cx - bw / 2.0) * width)))
    y1 = int(max(0, round((cy - bh / 2.0) * height)))
    x2 = int(min(width, round((cx + bw / 2.0) * width)))
    y2 = int(min(height, round((cy + bh / 2.0) * height)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def expand_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    scale: float,
    min_side: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = yolo_to_xyxy(box, width, height)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = min(max((x2 - x1) * scale, (y2 - y1) * scale, float(min_side)), float(max(width, height)))
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def find_original(image_dir: Path, generated: Path) -> Path | None:
    if not generated.stem.endswith("_gen"):
        return None
    stem = generated.stem[:-4]
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = [p for p in image_dir.glob(f"{stem}.*") if "_gen" not in p.stem]
    return matches[0] if matches else None


def gray(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def sobel_magnitude(g: np.ndarray) -> np.ndarray:
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    gx = (
        -p[:-2, :-2]
        - 2 * p[1:-1, :-2]
        - p[2:, :-2]
        + p[:-2, 2:]
        + 2 * p[1:-1, 2:]
        + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2]
        - 2 * p[:-2, 1:-1]
        - p[:-2, 2:]
        + p[2:, :-2]
        + 2 * p[2:, 1:-1]
        + p[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def edge_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    ma = sobel_magnitude(a)
    mb = sobel_magnitude(b)
    threshold = max(10.0, float(np.percentile(np.concatenate([ma.ravel(), mb.ravel()]), 82)))
    ea = ma >= threshold
    eb = mb >= threshold
    union = np.logical_or(ea, eb).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(ea, eb).sum()
    return float(1.0 - inter / union)


def largest_component_pct(mask: np.ndarray) -> float:
    if mask.size == 0 or not mask.any():
        return 0.0
    if cv2 is not None:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
        if num_labels <= 1:
            return 0.0
        best = int(stats[1:, cv2.CC_STAT_AREA].max())
        return float(best / mask.size * 100.0)
    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    best = 0
    ys, xs = np.where(mask)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        visited[sy, sx] = True
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, count)
    return float(best / mask.size * 100.0)


def score_pair(
    dataset: str,
    names: dict[int, str],
    original_path: Path,
    generated_path: Path,
    label_path: Path,
    class_id: int,
    box: tuple[float, float, float, float],
) -> Candidate | None:
    original = Image.open(original_path).convert("RGB")
    generated = Image.open(generated_path).convert("RGB")
    if original.size != generated.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
    oa = np.asarray(original)
    ga = np.asarray(generated)
    h, w = oa.shape[:2]
    x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    og = gray(oa[y1:y2, x1:x2])
    gg = gray(ga[y1:y2, x1:x2])
    diff = np.abs(gg - og)
    mean_abs = float(diff.mean())
    p95 = float(np.percentile(diff, 95))
    changed8 = float((diff > 8.0).mean() * 100.0)
    changed16 = float((diff > 16.0).mean() * 100.0)
    edge = edge_distance(og, gg)
    component = largest_component_pct(diff > max(8.0, p95 * 0.45))
    area_pct = float((x2 - x1) * (y2 - y1) / (w * h) * 100.0)

    score = 100.0 * (
        0.20 * min(mean_abs / 12.0, 1.0)
        + 0.20 * min(p95 / 35.0, 1.0)
        + 0.24 * min(changed8 / 35.0, 1.0)
        + 0.22 * min(edge, 1.0)
        + 0.14 * min(component / 25.0, 1.0)
    )

    return Candidate(
        dataset=dataset,
        class_id=class_id,
        class_name=names.get(class_id, str(class_id)),
        original_image=str(original_path),
        generated_image=str(generated_path),
        label=str(label_path),
        bbox_yolo=box,
        bbox_area_pct=round(area_pct, 4),
        mean_abs_diff=round(mean_abs, 4),
        p95_abs_diff=round(p95, 4),
        changed_pct_gt8=round(changed8, 4),
        changed_pct_gt16=round(changed16, 4),
        edge_jaccard_distance=round(edge, 4),
        largest_component_pct=round(component, 4),
        structure_score=round(score, 4),
    )


def collect_candidates(dataset: str, root: Path) -> list[Candidate]:
    names = load_names(root / "data.yaml")
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    candidates: list[Candidate] = []
    for generated in sorted(image_dir.glob("*_gen.*")):
        original = find_original(image_dir, generated)
        if original is None:
            continue
        label = label_dir / f"{generated.stem}.txt"
        boxes = read_boxes(label)
        best: Candidate | None = None
        for class_id, box in boxes:
            candidate = score_pair(dataset, names, original, generated, label, class_id, box)
            if candidate is not None and (best is None or candidate.structure_score > best.structure_score):
                best = candidate
        if best is not None:
            candidates.append(best)
    return candidates


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/calibri.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    out = Image.new("RGB", size, PANEL)
    ratio = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
    out.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return out


def diff_heatmap(original: Image.Image, generated: Image.Image, size: tuple[int, int]) -> Image.Image:
    oa = np.asarray(original.convert("RGB"), dtype=np.int16)
    ga = np.asarray(generated.convert("RGB").resize(original.size), dtype=np.int16)
    diff = np.abs(ga - oa).mean(axis=2).astype(np.float32)
    scale = np.percentile(diff, 98)
    if scale <= 0:
        scale = 1.0
    norm = np.clip(diff / scale, 0, 1)
    heat = np.zeros((*norm.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip((1 - np.abs(norm - 0.5) * 2) * 180, 0, 180).astype(np.uint8)
    heat[..., 2] = np.clip((1 - norm) * 80, 0, 80).astype(np.uint8)
    return fit_image(Image.fromarray(heat), size)


def draw_border(image: Image.Image, color: tuple[int, int, int], width: int = 4) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for i in range(width):
        draw.rectangle((i, i, image.width - 1 - i, image.height - 1 - i), outline=color)
    return image


def make_candidate_figure(candidates: list[Candidate], output: Path, title: str, max_rows: int, crop_scale: float, min_crop: int) -> None:
    candidates = candidates[:max_rows]
    title_font = font(27)
    header_font = font(20)
    body_font = font(15)
    tiny_font = font(13)
    panel_size = (250, 180)
    label_w = 250
    gap = 16
    top_h = 112
    row_h = 230
    columns = ["Original zoom", "DRFT-v2 zoom", "Difference heatmap"]
    width = label_w + len(columns) * panel_size[0] + (len(columns) + 1) * gap
    height = top_h + row_h * len(candidates) + gap
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 18), title, font=title_font, fill=TEXT)
    draw.text((gap, 56), "Ranked by bbox-local structural-change score: pixel change, edge mismatch, and changed connected area.", font=body_font, fill=MUTED)

    x = label_w + gap
    for col in columns:
        bbox = draw.textbbox((0, 0), col, font=header_font)
        draw.text((x + panel_size[0] // 2 - (bbox[2] - bbox[0]) // 2, top_h - 34), col, font=header_font, fill=TEXT)
        x += panel_size[0] + gap

    for row, cand in enumerate(candidates):
        y = top_h + row * row_h
        original = Image.open(cand.original_image).convert("RGB")
        generated = Image.open(cand.generated_image).convert("RGB")
        crop = expand_box(cand.bbox_yolo, original.width, original.height, crop_scale, min_crop)
        oz = original.crop(crop)
        gz = generated.crop(crop)
        panels = [
            draw_border(fit_image(oz, panel_size), BLUE),
            draw_border(fit_image(gz, panel_size), RED),
            draw_border(diff_heatmap(oz, gz, panel_size), (40, 40, 40), 2),
        ]

        draw.rounded_rectangle((gap, y + 12, label_w - gap, y + row_h - 12), radius=7, fill=PANEL, outline=(225, 225, 225))
        draw.text((gap + 12, y + 26), f"{row + 1}. {cand.dataset.upper()} / {cand.class_name}", font=header_font, fill=TEXT)
        draw.text((gap + 12, y + 58), Path(cand.generated_image).stem.replace("_gen", "")[:28], font=tiny_font, fill=MUTED)
        draw.text((gap + 12, y + 84), f"score {cand.structure_score:.1f} | diff {cand.mean_abs_diff:.1f} | p95 {cand.p95_abs_diff:.1f}", font=body_font, fill=TEXT)
        draw.text((gap + 12, y + 110), f"changed>8 {cand.changed_pct_gt8:.1f}% | edge {cand.edge_jaccard_distance:.2f}", font=body_font, fill=TEXT)
        draw.text((gap + 12, y + 136), f"bbox area {cand.bbox_area_pct:.2f}%", font=body_font, fill=MUTED)

        x = label_w + gap
        for panel in panels:
            canvas.paste(panel, (x, y + 24))
            x += panel_size[0] + gap

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def write_csv(path: Path, rows: Iterable[Candidate]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank generated samples by local defect structural change.")
    parser.add_argument("--datasets", default="gc10,mt,tilda")
    parser.add_argument("--output-dir", default="experiments/results/visualizations/structural_change_candidates")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--figure-rows", type=int, default=8)
    parser.add_argument("--crop-scale", type=float, default=3.0)
    parser.add_argument("--min-crop", type=int, default=90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_candidates: list[Candidate] = []
    by_dataset: dict[str, list[Candidate]] = {}
    for dataset in dataset_names:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        candidates = collect_candidates(dataset, DATASETS[dataset])
        candidates.sort(key=lambda c: c.structure_score, reverse=True)
        by_dataset[dataset] = candidates
        all_candidates.extend(candidates)
        write_csv(output_dir / f"{dataset}_structural_change_ranked.csv", candidates)
        (output_dir / f"{dataset}_structural_change_top{args.top_k}.json").write_text(
            json.dumps([asdict(c) for c in candidates[: args.top_k]], indent=2),
            encoding="utf-8",
        )
        make_candidate_figure(
            candidates,
            output_dir / f"{dataset}_structural_change_top{args.figure_rows}.png",
            f"{dataset.upper()} high structural-change candidates",
            args.figure_rows,
            args.crop_scale,
            args.min_crop,
        )

    all_candidates.sort(key=lambda c: c.structure_score, reverse=True)
    write_csv(output_dir / "all_structural_change_ranked.csv", all_candidates)
    (output_dir / f"all_structural_change_top{args.top_k}.json").write_text(
        json.dumps([asdict(c) for c in all_candidates[: args.top_k]], indent=2),
        encoding="utf-8",
    )
    make_candidate_figure(
        all_candidates,
        output_dir / f"all_structural_change_top{args.figure_rows}.png",
        "Top high structural-change candidates across datasets",
        args.figure_rows,
        args.crop_scale,
        args.min_crop,
    )

    summary = {
        "datasets": {
            name: {
                "num_generated_pairs": len(rows),
                "top_score": rows[0].structure_score if rows else None,
                "top_sample": asdict(rows[0]) if rows else None,
            }
            for name, rows in by_dataset.items()
        },
        "top_overall": [asdict(c) for c in all_candidates[: min(10, len(all_candidates))]],
    }
    (output_dir / "structural_change_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)
    print(output_dir / "structural_change_summary.json")


if __name__ == "__main__":
    main()
