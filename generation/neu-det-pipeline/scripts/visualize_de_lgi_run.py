from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neu_det_pipeline.data.loader import DefectSample, collect_dataset_instances  # noqa: E402


def _load_rows(score_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in score_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("selected"):
            rows.append(row)
    return rows


def _sample_index(samples: list[DefectSample]) -> dict[tuple[str, int], DefectSample]:
    indexed: dict[tuple[str, int], DefectSample] = {}
    for sample in samples:
        indexed[(sample.source_stem, int(sample.object_index))] = sample
    return indexed


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _diff_image(real: Image.Image, generated: Image.Image, size: tuple[int, int]) -> Image.Image:
    real_arr = np.asarray(real.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    gen_arr = np.asarray(generated.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    diff = np.abs(gen_arr - real_arr).mean(axis=2)
    if float(diff.max()) > 1e-6:
        diff = diff / float(diff.max())
    heat = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    heat[..., 0] = np.clip(diff * 255.0, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip((1.0 - diff) * 70.0, 0, 70).astype(np.uint8)
    heat[..., 2] = np.clip((1.0 - diff) * 45.0, 0, 45).astype(np.uint8)
    return Image.fromarray(heat, mode="RGB")


def _scale_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    sx = target_size[0] / max(1, source_size[0])
    sy = target_size[1] / max(1, source_size[1])
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))


def _safe_bbox(value: Any, fallback: list[int] | tuple[int, int, int, int]) -> list[int] | tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in value[:4]]
        except (TypeError, ValueError):
            return fallback
        if x1 > x0 and y1 > y0:
            return [x0, y0, x1, y1]
    return fallback


def _generated_image_path(row: dict[str, Any]) -> Path | None:
    raw = row.get("output_path") or row.get("candidate_path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font: ImageFont.ImageFont) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    clipped = text
    while clipped and draw.textlength(clipped + suffix, font=font) > max_width:
        clipped = clipped[:-1]
    return (clipped + suffix) if clipped else suffix


def _draw_panel(
    sample: DefectSample,
    row: dict[str, Any],
    *,
    cell_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> Image.Image:
    header_h = 42
    panel = Image.new("RGB", (cell_size[0] * 3, cell_size[1] + header_h), "white")
    draw = ImageDraw.Draw(panel)
    real = Image.open(sample.image_path).convert("RGB")
    generated_path = _generated_image_path(row)
    if generated_path is None:
        raise FileNotFoundError(row.get("output_path") or row.get("candidate_path"))
    generated = Image.open(generated_path).convert("RGB")
    real_fit = _fit_image(real, cell_size)
    gen_fit = _fit_image(generated, cell_size)
    diff_fit = _diff_image(real, generated, cell_size)
    panel.paste(real_fit, (0, header_h))
    panel.paste(gen_fit, (cell_size[0], header_h))
    panel.paste(diff_fit, (cell_size[0] * 2, header_h))

    real_bbox = _scale_bbox(sample.bbox, real.size, cell_size)
    spatial_code = row.get("spatial_code", {})
    gen_bbox_raw = spatial_code.get("generated_bbox") if isinstance(spatial_code, dict) else None
    gen_bbox = _safe_bbox(gen_bbox_raw, sample.bbox)
    gen_bbox_scaled = _scale_bbox(gen_bbox, generated.size, cell_size)
    real_bbox = (real_bbox[0], real_bbox[1] + header_h, real_bbox[2], real_bbox[3] + header_h)
    draw.rectangle(real_bbox, outline="red", width=2)
    gx0, gy0, gx1, gy1 = gen_bbox_scaled
    draw.rectangle((gx0 + cell_size[0], gy0 + header_h, gx1 + cell_size[0], gy1 + header_h), outline="red", width=2)
    draw.rectangle((gx0 + cell_size[0] * 2, gy0 + header_h, gx1 + cell_size[0] * 2, gy1 + header_h), outline="yellow", width=2)

    quality = row.get("quality", {})
    quality = quality if isinstance(quality, dict) else {}
    label = row.get("label_contract", "")
    s2i_score = _safe_float(quality.get("s2i_score"))
    s2c_score = _safe_float(quality.get("hdsi_s2c_score"))
    inside_delta = _safe_float(quality.get("inside_delta"))
    title = f"{row.get('class', '')} | s2c={s2c_score:.2f} | s2i={s2i_score:.2f} | in={inside_delta:.3f}"
    draw.text((4, 4), _clip_text(draw, title, cell_size[0] * 3 - 8, font), fill="black", font=font)
    draw.text((4, 23), _clip_text(draw, f"real | {sample.source_stem}", cell_size[0] - 8, font), fill="black", font=font)
    draw.text(
        (cell_size[0] + 4, 23),
        _clip_text(draw, f"generated | {label}", cell_size[0] - 8, font),
        fill="black",
        font=font,
    )
    draw.text((cell_size[0] * 2 + 4, 23), "abs diff", fill="black", font=font)
    return panel


def _make_grid(panels: list[Image.Image], columns: int, gap: int = 10) -> Image.Image:
    if not panels:
        return Image.new("RGB", (800, 120), "white")
    w, h = panels[0].size
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * w + (columns - 1) * gap, rows * h + (rows - 1) * gap), "white")
    for idx, panel in enumerate(panels):
        x = (idx % columns) * (w + gap)
        y = (idx // columns) * (h + gap)
        canvas.paste(panel, (x, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize selected DE-LGI/SRT generated samples.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--detail-count", type=int, default=12)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    score_path = run_dir / "artifacts" / "de_lgi" / "candidate_scores.jsonl"
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    rows = _load_rows(score_path)
    samples = _sample_index(collect_dataset_instances(args.dataset_root))
    matched: list[tuple[DefectSample, dict[str, Any]]] = []
    for row in rows:
        try:
            key = (str(row.get("source_key", "")), int(row.get("object_index", 0)))
        except (TypeError, ValueError):
            continue
        sample = samples.get(key)
        if sample is not None and _generated_image_path(row) is not None:
            matched.append((sample, row))

    out_dir = (args.out_dir or (run_dir / "visualizations")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    detail_panels = [
        _draw_panel(sample, row, cell_size=(176, 176), font=font)
        for sample, row in matched[: max(1, int(args.detail_count))]
    ]
    overview_panels = [
        _draw_panel(sample, row, cell_size=(104, 104), font=font)
        for sample, row in matched
    ]
    detail = _make_grid(detail_panels, columns=3)
    overview = _make_grid(overview_panels, columns=5)
    detail_path = out_dir / "real_vs_generated_detail.png"
    overview_path = out_dir / "real_vs_generated_overview.png"
    detail.save(detail_path)
    overview.save(overview_path)
    summary = {
        "run_dir": str(run_dir),
        "score_path": str(score_path),
        "selected_rows": len(rows),
        "matched_rows": len(matched),
        "detail": str(detail_path),
        "overview": str(overview_path),
    }
    (out_dir / "visualization_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
