from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover_score_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "artifacts" / "dbt_cmdp" / "candidate_scores.jsonl",
        run_dir / "artifacts" / "reference_cmdp" / "candidate_scores.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(run_dir.rglob("candidate_scores.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No candidate_scores.jsonl found under {run_dir}")
    return matches[0]


def _find_mixed_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "mixed_dataset" / "mixed_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    matches = sorted(run_dir.rglob("mixed_manifest.json"))
    if matches:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    return None


def _numbers(rows: list[dict[str, Any]], dotted_key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        cur: Any = row
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if isinstance(cur, (int, float)) and math.isfinite(float(cur)):
            values.append(float(cur))
    return values


def _stat(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _flat_get(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        cur: Any = row
        ok = True
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                ok = False
                break
            cur = cur[part]
        if ok:
            return cur
    return None


def _image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, path)
    return index


def _candidate_path(row: dict[str, Any], run_dir: Path) -> Path | None:
    raw = row.get("candidate_path") or _flat_get(row, ["cmdp.output_path"])
    if isinstance(raw, str):
        path = Path(raw)
        if path.exists():
            return path
    target = row.get("target_key")
    if isinstance(target, str):
        for suffix in [".png", ".jpg", ".jpeg"]:
            path = run_dir / "images" / f"{target}{suffix}"
            if path.exists():
                return path
    return None


def _defect_box(row: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    expanded = _flat_get(row, ["cmdp.expanded_bbox"])
    local = _flat_get(row, ["cmdp.defect_bbox_in_expanded"])
    if (
        isinstance(expanded, list)
        and isinstance(local, list)
        and len(expanded) == 4
        and len(local) == 4
    ):
        x1 = int(round(float(expanded[0]) + float(local[0])))
        y1 = int(round(float(expanded[1]) + float(local[1])))
        x2 = int(round(float(expanded[0]) + float(local[2])))
        y2 = int(round(float(expanded[1]) + float(local[3])))
        return _clip_box((x1, y1, x2, y2), width, height)
    return None


def _clip_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _expand_box(
    box: tuple[int, int, int, int], width: int, height: int, scale: float, min_size: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(float(x2 - x1) * scale, float(min_size))
    bh = max(float(y2 - y1) * scale, float(min_size))
    out = (
        int(round(cx - bw / 2.0)),
        int(round(cy - bh / 2.0)),
        int(round(cx + bw / 2.0)),
        int(round(cy + bh / 2.0)),
    )
    clipped = _clip_box(out, width, height)
    if clipped is None:
        return 0, 0, width, height
    return clipped


def _panel(image: Image.Image, title: str, size: int = 192) -> Image.Image:
    canvas = Image.new("RGB", (size, size + 24), "white")
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    x = (size - image.width) // 2
    y = 24 + (size - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 5), title[:34], fill=(20, 20, 20), font=font)
    return canvas


def _make_diff(a: Image.Image, b: Image.Image) -> Image.Image:
    b = b.resize(a.size, Image.Resampling.BILINEAR)
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    return ImageEnhance.Contrast(diff).enhance(3.0)


def _write_zoom_grid(
    rows: list[dict[str, Any]],
    run_dir: Path,
    original_images_root: Path | None,
    out_path: Path,
    samples_per_class: int,
    crop_scale: float,
    min_crop_size: int,
) -> int:
    if original_images_root is None:
        return 0
    original_index = _image_index(original_images_root)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cls_name = str(row.get("class") or _flat_get(row, ["cmdp.class_name"]) or "unknown")
        by_class[cls_name].append(row)

    selected: list[dict[str, Any]] = []
    for cls_name in sorted(by_class):
        cls_rows = sorted(
            by_class[cls_name],
            key=lambda item: float(_flat_get(item, ["quality.total"]) or -1.0),
            reverse=True,
        )
        selected.extend(cls_rows[:samples_per_class])

    strips: list[Image.Image] = []
    for row in selected:
        source_key = row.get("source_key")
        if not isinstance(source_key, str):
            continue
        original_path = original_index.get(source_key)
        gen_path = _candidate_path(row, run_dir)
        if original_path is None or gen_path is None:
            continue
        try:
            with Image.open(original_path) as original_raw, Image.open(gen_path) as generated_raw:
                original = original_raw.convert("RGB")
                generated = generated_raw.convert("RGB")
                width, height = original.size
                box = _defect_box(row, width, height)
                if box is None:
                    box = (0, 0, width, height)
                crop_box = _expand_box(box, width, height, crop_scale, min_crop_size)
                original_crop = original.crop(crop_box)
                if generated.size != original.size:
                    generated = generated.resize(original.size, Image.Resampling.BILINEAR)
                generated_crop = generated.crop(crop_box)
                diff_crop = _make_diff(original_crop, generated_crop)
        except OSError:
            continue

        cls_name = str(row.get("class") or "unknown")
        q = _flat_get(row, ["quality.total"])
        q_text = f"{float(q):.3f}" if isinstance(q, (int, float)) else "na"
        panels = [
            _panel(original_crop, f"{cls_name} original"),
            _panel(generated_crop, f"generated q={q_text}"),
            _panel(diff_crop, "enhanced diff"),
        ]
        strip = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "white")
        x = 0
        for panel in panels:
            strip.paste(panel, (x, 0))
            x += panel.width
        strips.append(strip)

    if not strips:
        return 0
    gap = 8
    grid = Image.new(
        "RGB",
        (max(strip.width for strip in strips), sum(strip.height for strip in strips) + gap * (len(strips) - 1)),
        "white",
    )
    y = 0
    for strip in strips:
        grid.paste(strip, (0, y))
        y += strip.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return len(strips)


def summarize(run_dir: Path, dataset_root: Path | None, output_dir: Path | None, samples_per_class: int) -> dict[str, Any]:
    score_path = _discover_score_path(run_dir)
    rows = _read_jsonl(score_path)
    mixed_manifest = _find_mixed_manifest(run_dir)
    if output_dir is None:
        output_dir = score_path.parent / "review"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = [row for row in rows if row.get("selected", True)]
    class_counts = Counter(str(row.get("class") or _flat_get(row, ["cmdp.class_name"]) or "unknown") for row in rows)
    selected_class_counts = Counter(
        str(row.get("class") or _flat_get(row, ["cmdp.class_name"]) or "unknown") for row in selected_rows
    )
    family_counts = Counter(
        str(_flat_get(row, ["dbt.stochastic_family"]) or "unset") for row in rows
    )
    prompt_counts = Counter(str(row.get("prompt") or "") for row in rows)

    numeric_keys = [
        "quality.total",
        "quality.background_preservation",
        "quality.boundary_score",
        "quality.outside_delta",
        "cmdp.pseudo_area_ratio",
        "cmdp.core_area_ratio",
        "cmdp.band_area_ratio",
        "cmdp.eta_mean",
        "dbt.band_radius",
        "dbt.residual_blur_sigma",
        "dbt.gradient_weight",
        "dbt.max_area_ratio",
        "dbt.box_inner_strength",
        "dbt.box_pseudo_strength",
        "dbt.box_boundary_strength",
        "dbt.mask_stats.pseudo_area_ratio",
        "dbt.mask_stats.box_area_ratio",
        "dbt.mask_stats.core_area_ratio",
        "dbt.mask_stats.band_area_ratio",
    ]
    numeric_summary = {key: _stat(_numbers(rows, key)) for key in numeric_keys}
    param_keys = [
        "dbt.band_radius",
        "dbt.residual_blur_sigma",
        "dbt.gradient_weight",
        "dbt.max_area_ratio",
        "dbt.box_inner_strength",
        "dbt.box_pseudo_strength",
        "dbt.box_boundary_strength",
    ]
    param_unique = {
        key: len({round(value, 4) for value in _numbers(rows, key)})
        for key in param_keys
    }

    source_count = len({str(row.get("source_key")) for row in rows if row.get("source_key")})
    target_count = len({str(row.get("target_key")) for row in rows if row.get("target_key")})
    summary = {
        "run_dir": str(run_dir.resolve()),
        "score_path": str(score_path.resolve()),
        "total_rows": len(rows),
        "selected_rows": len(selected_rows),
        "source_count": source_count,
        "target_count": target_count,
        "class_counts": dict(sorted(class_counts.items())),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "stochastic_box_rows": sum(1 for row in rows if _flat_get(row, ["dbt.stochastic_box"]) is True),
        "stochastic_profiles": dict(
            sorted(Counter(str(_flat_get(row, ["dbt.stochastic_profile"]) or "unset") for row in rows).items())
        ),
        "stochastic_families": dict(sorted(family_counts.items())),
        "unique_prompts": len([key for key in prompt_counts if key]),
        "numeric": numeric_summary,
        "parameter_unique_values": param_unique,
        "mixed_manifest": mixed_manifest,
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_class_path = output_dir / "per_class_summary.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class",
                "rows",
                "quality_mean",
                "outside_delta_mean",
                "pseudo_area_mean",
                "core_area_mean",
                "band_area_mean",
            ],
        )
        writer.writeheader()
        for cls_name in sorted(class_counts):
            cls_rows = [
                row
                for row in rows
                if str(row.get("class") or _flat_get(row, ["cmdp.class_name"]) or "unknown") == cls_name
            ]
            writer.writerow(
                {
                    "class": cls_name,
                    "rows": len(cls_rows),
                    "quality_mean": _stat(_numbers(cls_rows, "quality.total"))["mean"],
                    "outside_delta_mean": _stat(_numbers(cls_rows, "quality.outside_delta"))["mean"],
                    "pseudo_area_mean": _stat(_numbers(cls_rows, "cmdp.pseudo_area_ratio"))["mean"],
                    "core_area_mean": _stat(_numbers(cls_rows, "cmdp.core_area_ratio"))["mean"],
                    "band_area_mean": _stat(_numbers(cls_rows, "cmdp.band_area_ratio"))["mean"],
                }
            )

    original_root: Path | None = None
    if dataset_root is not None:
        original_root = dataset_root / "images"
    elif isinstance(mixed_manifest, dict) and isinstance(mixed_manifest.get("orig_images_dir"), str):
        original_root = Path(str(mixed_manifest["orig_images_dir"]))

    zoom_count = _write_zoom_grid(
        rows,
        run_dir,
        original_root,
        output_dir / "zoom_compare_grid.png",
        samples_per_class=samples_per_class,
        crop_scale=2.5,
        min_crop_size=72,
    )
    summary["review_outputs"] = {
        "generation_summary_json": str((output_dir / "generation_summary.json").resolve()),
        "per_class_summary_csv": str(per_class_path.resolve()),
        "zoom_compare_grid": str((output_dir / "zoom_compare_grid.png").resolve()) if zoom_count else None,
        "zoom_rows": zoom_count,
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a DBT-CMDP run and create zoom comparisons.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples-per-class", type=int, default=3)
    args = parser.parse_args()
    summary = summarize(
        args.run_dir,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        samples_per_class=max(1, args.samples_per_class),
    )
    print(json.dumps(summary["review_outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
