#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_NEU_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def load_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _strip_yaml_scalar(value: str) -> str:
    value = value.strip().strip(",")
    if not value:
        return value
    if (value[0], value[-1]) in {("'", "'"), ('"', '"')}:
        return value[1:-1]
    return value


def _read_yolo_names(data_yaml: Path) -> list[str] | None:
    if not data_yaml.exists():
        return None
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    names: dict[int, str] = {}
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        inline = stripped.split(":", 1)[1].strip()
        if inline:
            try:
                parsed = ast.literal_eval(inline)
                if isinstance(parsed, list) and parsed:
                    return [str(item) for item in parsed]
                if isinstance(parsed, dict) and parsed:
                    return [str(parsed[key]) for key in sorted(parsed)]
            except Exception:
                return None
        base_indent = len(line) - len(line.lstrip())
        for child in lines[idx + 1 :]:
            if not child.strip() or child.lstrip().startswith("#"):
                continue
            indent = len(child) - len(child.lstrip())
            if indent <= base_indent:
                break
            match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", child)
            if match:
                names[int(match.group(1))] = _strip_yaml_scalar(match.group(2))
        break
    if names:
        return [names[key] for key in sorted(names)]
    return None


def _infer_yolo_names(orig_images_dir: Path, orig_labels_dir: Path) -> list[str]:
    candidates: list[Path] = []
    for root in (orig_labels_dir, orig_labels_dir.parent, orig_images_dir, orig_images_dir.parent):
        candidates.append(root / "data.yaml")
        candidates.append(root.parent / "data.yaml")
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        names = _read_yolo_names(candidate)
        if names:
            return names
    return list(DEFAULT_NEU_NAMES)


def _yaml_name(value: str) -> str:
    if re.match(r"^[A-Za-z0-9_.-]+$", value):
        return value
    return json.dumps(value, ensure_ascii=False)


def copy_image_and_label(src_img: Path, src_labels_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path) -> None:
    stem = Path(src_img).stem
    lbl_file = src_labels_dir / f"{stem}.txt"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img_dir / Path(src_img).name)
    if lbl_file.exists():
        shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)


def _bbox_to_yolo_line(cls_id: int, bbox: list[float], image_size: tuple[int, int]) -> str:
    width, height = image_size
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0 = max(0.0, min(float(width), x0))
    x1 = max(0.0, min(float(width), x1))
    y0 = max(0.0, min(float(height), y0))
    y1 = max(0.0, min(float(height), y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    xc = (x0 + x1) * 0.5 / max(1.0, float(width))
    yc = (y0 + y1) * 0.5 / max(1.0, float(height))
    wn = bw / max(1.0, float(width))
    hn = bh / max(1.0, float(height))
    return f"{int(cls_id)} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def _yolo_line_to_pixel_bbox(line: str, image_size: tuple[int, int]) -> tuple[int, list[float]] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = [float(v) for v in parts[1:5]]
    except ValueError:
        return None
    width, height = image_size
    x0 = (xc - bw * 0.5) * width
    y0 = (yc - bh * 0.5) * height
    x1 = (xc + bw * 0.5) * width
    y1 = (yc + bh * 0.5) * height
    return cls_id, [x0, y0, x1, y1]


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-6, area_a + area_b - inter)


def _generated_label_lines(
    row: dict[str, Any] | None,
    *,
    class_names: list[str],
    image_size: tuple[int, int],
    fallback_label_path: Path,
) -> list[str] | None:
    if not isinstance(row, dict):
        return None
    spatial = row.get("spatial_code")
    if not isinstance(spatial, dict):
        return None
    generated_bbox = spatial.get("generated_bbox")
    if not isinstance(generated_bbox, list) or len(generated_bbox) != 4:
        return None
    cls_name = _candidate_class(row)
    cls_id = class_names.index(cls_name) if cls_name in class_names else None
    fallback_lines = []
    if fallback_label_path.exists():
        fallback_lines = [line.strip() for line in fallback_label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if cls_id is None and fallback_lines:
        parsed = _yolo_line_to_pixel_bbox(fallback_lines[0], image_size)
        if parsed is not None:
            cls_id = parsed[0]
    if cls_id is None:
        return None
    generated_line = _bbox_to_yolo_line(int(cls_id), [float(v) for v in generated_bbox], image_size)
    if not fallback_lines:
        return [generated_line]

    source_bbox = spatial.get("source_bbox")
    best_idx = None
    best_score = -1.0
    if isinstance(source_bbox, list) and len(source_bbox) == 4:
        for idx, line in enumerate(fallback_lines):
            parsed = _yolo_line_to_pixel_bbox(line, image_size)
            if parsed is None:
                continue
            label_cls, bbox = parsed
            class_bonus = 0.25 if label_cls == int(cls_id) else 0.0
            score = _bbox_iou([float(v) for v in source_bbox], bbox) + class_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
    if best_idx is None:
        for idx, line in enumerate(fallback_lines):
            parsed = _yolo_line_to_pixel_bbox(line, image_size)
            if parsed is not None and parsed[0] == int(cls_id):
                best_idx = idx
                break
    lines = list(fallback_lines)
    if best_idx is None:
        lines.append(generated_line)
    else:
        lines[best_idx] = generated_line
    return lines


def find_original_image(orig_images_dir: Path, stem: str, split: str | None = None) -> Path | None:
    roots = []
    if split:
        roots.append(orig_images_dir / split)
    roots.append(orig_images_dir)
    for root in roots:
        for suffix in IMAGE_SUFFIXES:
            candidate = root / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def source_stem_for_generated(stem: str, known_stems: set[str]) -> str:
    """Map instance-level generated names like sample_o03 back to sample."""

    if stem in known_stems:
        return stem
    match = re.match(r"^(?P<source>.+)_o\d{2,}$", stem)
    if match and match.group("source") in known_stems:
        return match.group("source")
    return stem


@dataclass
class GeneratedCandidate:
    image: Path
    gen_input_stem: str
    source_stem: str
    split: str
    label_path: Path
    score_row: dict[str, Any] | None


def _nested_float(row: dict[str, Any] | None, path: str, default: float | None = None) -> float | None:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
    if isinstance(current, (int, float)):
        return float(current)
    return default


def _nested_bool(row: dict[str, Any] | None, path: str, default: bool | None = None) -> bool | None:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
    if isinstance(current, bool):
        return current
    if isinstance(current, str):
        value = current.strip().lower()
        if value in {"true", "1", "yes"}:
            return True
        if value in {"false", "0", "no"}:
            return False
    return default


def _candidate_score(row: dict[str, Any] | None) -> float:
    selection_score = _nested_float(row, "selection_score")
    if selection_score is not None:
        return selection_score
    return _nested_float(row, "quality.total", 0.0) or 0.0


def _candidate_utility_score(row: dict[str, Any] | None) -> float:
    selection_score = _nested_float(row, "selection_score")
    if selection_score is not None:
        return selection_score
    quality_score = _candidate_score(row)
    reliable_score = _nested_float(row, "reliable_quality.total")
    if reliable_score is None:
        return quality_score
    context_score = _nested_float(row, "context_quality.total", 1.0) or 1.0
    return float(0.42 * quality_score + 0.40 * reliable_score + 0.18 * context_score)


def _candidate_class(row: dict[str, Any] | None) -> str:
    if isinstance(row, dict):
        value = row.get("class") or row.get("cls_name")
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _candidate_annotation_gates_pass(row: dict[str, Any] | None) -> bool:
    if row is None:
        return True
    accepted = _nested_bool(row, "candidate_accepted")
    if accepted is False:
        return False
    label_contract = row.get("label_contract")
    if isinstance(label_contract, str) and label_contract == "structure-evidence-failed":
        return False
    s2i_pass = _nested_bool(row, "quality.s2i_gate_pass")
    s2i_strict = _nested_bool(row, "latent_inpainting.srt_strict_s2i_gate")
    if s2i_pass is False and s2i_strict is not False:
        return False
    seca_pass = _nested_bool(row, "quality.seca_gate_pass")
    if seca_pass is False:
        return False
    return True


def _load_score_index(score_path: Path | None) -> dict[str, dict[str, Any]]:
    if score_path is None:
        return {}
    score_path = Path(score_path)
    if not score_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in score_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("target_key") or row.get("stem")
        if not isinstance(key, str) or not key:
            continue
        # Prefer the candidate selected by the generator when multiple rows
        # exist for the same target.
        if key not in rows or bool(row.get("selected")):
            rows[key] = row
    return rows


def _passes_quality_filters(
    row: dict[str, Any] | None,
    *,
    quality_threshold: float | None,
    min_canvas_drop: float | None,
    max_outside_delta: float | None,
) -> bool:
    if row is None:
        return True
    if not _candidate_annotation_gates_pass(row):
        return False
    if quality_threshold is not None and _candidate_utility_score(row) < float(quality_threshold):
        return False
    if min_canvas_drop is not None:
        drop = _nested_float(row, "canvas.quality.defect_evidence_drop")
        if drop is not None and drop < float(min_canvas_drop):
            return False
    if max_outside_delta is not None:
        outside = _nested_float(row, "quality.outside_delta")
        if outside is not None and outside > float(max_outside_delta):
            return False
    return True


def _order_candidates(
    candidates: list[GeneratedCandidate],
    *,
    rng: random.Random,
    selection_strategy: str,
) -> list[GeneratedCandidate]:
    strategy = selection_strategy.lower().replace("_", "-")
    if strategy in {"steel-utility", "utility"}:
        return sorted(candidates, key=lambda item: (_candidate_utility_score(item.score_row), item.gen_input_stem), reverse=True)
    if strategy == "quality":
        return sorted(candidates, key=lambda item: (_candidate_score(item.score_row), item.gen_input_stem), reverse=True)
    if strategy in {"balanced-quality", "balanced-steel-utility", "balanced-utility"}:
        score_fn = _candidate_utility_score if "utility" in strategy else _candidate_score
        groups: dict[str, list[GeneratedCandidate]] = {}
        for item in candidates:
            groups.setdefault(_candidate_class(item.score_row), []).append(item)
        for cls_items in groups.values():
            cls_items.sort(key=lambda item: (score_fn(item.score_row), item.gen_input_stem), reverse=True)
        ordered: list[GeneratedCandidate] = []
        class_names = sorted(groups)
        cursor = 0
        while any(groups.values()):
            cls_name = class_names[cursor % len(class_names)]
            if groups[cls_name]:
                ordered.append(groups[cls_name].pop(0))
            cursor += 1
        return ordered
    rng.shuffle(candidates)
    return candidates


def _select_generated_candidates(
    candidates: list[GeneratedCandidate],
    *,
    rng: random.Random,
    max_generated: int | None,
    per_source_limit: int,
    per_class_limit: int,
    selection_strategy: str,
) -> tuple[list[GeneratedCandidate], dict[str, int]]:
    ordered = _order_candidates(candidates, rng=rng, selection_strategy=selection_strategy)
    selected: list[GeneratedCandidate] = []
    source_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    skipped = {"source_cap": 0, "class_cap": 0, "total_cap": 0}
    for item in ordered:
        if max_generated is not None and len(selected) >= int(max_generated):
            skipped["total_cap"] += 1
            continue
        if per_source_limit > 0 and source_counts.get(item.source_stem, 0) >= per_source_limit:
            skipped["source_cap"] += 1
            continue
        cls_name = _candidate_class(item.score_row)
        if per_class_limit > 0 and class_counts.get(cls_name, 0) >= per_class_limit:
            skipped["class_cap"] += 1
            continue
        selected.append(item)
        source_counts[item.source_stem] = source_counts.get(item.source_stem, 0) + 1
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    return selected, skipped


def create_mixed_dataset(
    orig_images_dir,
    orig_labels_dir,
    manifest_path,
    new_images_dir,
    run_output_dir=None,
    seed: int = 42,
    score_path=None,
    max_generated: int | None = None,
    per_source_limit: int = 0,
    per_class_limit: int = 0,
    quality_threshold: float | None = None,
    min_canvas_drop: float | None = None,
    max_outside_delta: float | None = None,
    selection_strategy: str = "random",
):
    rng = random.Random(seed)
    orig_images_dir = Path(orig_images_dir)
    orig_labels_dir = Path(orig_labels_dir)
    manifest_path = Path(manifest_path)
    new_images_dir = Path(new_images_dir)

    if run_output_dir is None or str(run_output_dir).strip() == "":
        run_output_dir = new_images_dir / "mixed_dataset"
    else:
        run_output_dir = Path(run_output_dir)

    manifest = load_manifest(manifest_path)

    out_images_train = run_output_dir / "images" / "train"
    out_images_val = run_output_dir / "images" / "val"
    out_images_test = run_output_dir / "images" / "test"
    out_labels_train = run_output_dir / "labels" / "train"
    out_labels_val = run_output_dir / "labels" / "val"
    out_labels_test = run_output_dir / "labels" / "test"

    stem_to_split: dict[str, str] = {}
    train_orig_count = 0
    for split, (out_img_dir, out_lbl_dir) in {
        "train": (out_images_train, out_labels_train),
        "val": (out_images_val, out_labels_val),
        "test": (out_images_test, out_labels_test),
    }.items():
        for stem in manifest.get(split, []):
            stem_to_split[stem] = split
            img_file = find_original_image(orig_images_dir, stem, split)
            if img_file is None:
                print(f"[WARN] Original image missing for {stem} in {split}")
                continue
            src_labels_dir = orig_labels_dir / split
            copy_image_and_label(img_file, src_labels_dir, out_img_dir, out_lbl_dir)
            if split == "train":
                train_orig_count += 1
    known_stems = set(stem_to_split)

    images_root = new_images_dir / "images" if (new_images_dir / "images").exists() else new_images_dir
    new_imgs = sorted([p for p in images_root.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    score_index = _load_score_index(Path(score_path) if score_path else None)
    score_filter_requested = score_path is not None

    print(f"[INFO] Found {len(new_imgs)} generated images")
    print(f"[INFO] Label directory: {orig_labels_dir}")
    if not orig_labels_dir.exists():
        print(f"[ERROR] Label directory does not exist: {orig_labels_dir}")
    class_names = _infer_yolo_names(orig_images_dir, orig_labels_dir)

    generated_count = 0
    skipped_no_manifest = 0
    skipped_no_label = 0
    skipped_data_leak = 0
    skipped_quality = 0
    candidates: list[GeneratedCandidate] = []
    for img in new_imgs:
        gen_input_stem = img.stem
        score_row = score_index.get(gen_input_stem)
        if score_filter_requested and score_row is None:
            skipped_quality += 1
            if skipped_quality <= 5:
                print(f"[WARN] Generated image {gen_input_stem} has no candidate score row, skipping")
            continue
        scored_source = score_row.get("source_key") if isinstance(score_row, dict) else None
        source_stem = scored_source if isinstance(scored_source, str) and scored_source else source_stem_for_generated(gen_input_stem, known_stems)
        split = stem_to_split.get(source_stem)
        if split is None:
            skipped_no_manifest += 1
            if skipped_no_manifest <= 5:
                print(f"[WARN] Generated image {gen_input_stem} not found in manifest, skipping")
            continue
        if split != "train":
            skipped_data_leak += 1
            if skipped_data_leak <= 5:
                print(f"[INFO] Skipping generated image {gen_input_stem} (original in {split}) to prevent data leakage")
            continue

        label_dir = orig_labels_dir / split
        lbl_src = label_dir / f"{source_stem}.txt"
        if not lbl_src.exists():
            skipped_no_label += 1
            if skipped_no_label <= 5:
                print(f"[WARN] Missing label for {source_stem} in {split} (expected: {lbl_src})")
            continue
        if not _passes_quality_filters(
            score_row,
            quality_threshold=quality_threshold,
            min_canvas_drop=min_canvas_drop,
            max_outside_delta=max_outside_delta,
        ):
            skipped_quality += 1
            continue

        candidates.append(
            GeneratedCandidate(
                image=img,
                gen_input_stem=gen_input_stem,
                source_stem=source_stem,
                split=split,
                label_path=lbl_src,
                score_row=score_row,
            )
        )

    selected, selection_skips = _select_generated_candidates(
        candidates,
        rng=rng,
        max_generated=max_generated,
        per_source_limit=max(0, int(per_source_limit)),
        per_class_limit=max(0, int(per_class_limit)),
        selection_strategy=selection_strategy,
    )

    for item in selected:
        gen_stem = f"{item.gen_input_stem}_gen"
        original_ext = item.image.suffix
        dst_img_path = out_images_train / f"{gen_stem}{original_ext}"
        out_images_train.mkdir(parents=True, exist_ok=True)
        with Image.open(item.image) as im:
            im = im.convert("RGB")
            im = im.resize((200, 200), Image.BILINEAR)
            if original_ext.lower() == ".png":
                im.save(dst_img_path, "PNG")
            else:
                im.save(dst_img_path, "JPEG", quality=95)

        out_labels_train.mkdir(parents=True, exist_ok=True)
        dst_label_path = out_labels_train / f"{gen_stem}.txt"
        with Image.open(item.image) as im:
            generated_label_lines = _generated_label_lines(
                item.score_row,
                class_names=class_names,
                image_size=im.size,
                fallback_label_path=item.label_path,
            )
        if generated_label_lines is None:
            shutil.copy2(item.label_path, dst_label_path)
        else:
            dst_label_path.write_text("\n".join(generated_label_lines) + "\n", encoding="utf-8")
        generated_count += 1

    if skipped_no_manifest:
        print(f"[WARN] Skipped {skipped_no_manifest} generated images missing from manifest")
    if skipped_data_leak:
        print(f"[INFO] Skipped {skipped_data_leak} generated images to prevent val/test leakage")
    if skipped_no_label:
        print(f"[WARN] Skipped {skipped_no_label} generated images without labels")
    if skipped_quality:
        print(f"[INFO] Skipped {skipped_quality} generated images by quality filters")
    for key, count in selection_skips.items():
        if count:
            print(f"[INFO] Skipped {count} generated images by {key.replace('_', ' ')}")

    mixed_manifest = {
        "seed": seed,
        "orig_images_dir": str(orig_images_dir),
        "orig_labels_dir": str(orig_labels_dir),
        "manifest_path": str(manifest_path),
        "new_images_dir": str(new_images_dir),
        "class_names": class_names,
        "train_original_count": train_orig_count,
        "train_generated_count": generated_count,
        "train_total_count": train_orig_count + generated_count,
        "val_count": len(list(out_images_val.glob("*.*"))),
        "test_count": len(list(out_images_test.glob("*.*"))),
        "selection": {
            "score_path": str(score_path) if score_path else None,
            "selection_strategy": selection_strategy,
            "max_generated": max_generated,
            "per_source_limit": per_source_limit,
            "per_class_limit": per_class_limit,
            "quality_threshold": quality_threshold,
            "min_canvas_drop": min_canvas_drop,
            "max_outside_delta": max_outside_delta,
            "candidate_count_after_split_label_quality": len(candidates),
            "skipped_quality": skipped_quality,
            **{f"skipped_{key}": value for key, value in selection_skips.items()},
        },
    }
    run_output_dir.mkdir(parents=True, exist_ok=True)
    (run_output_dir / "mixed_manifest.json").write_text(
        json.dumps(mixed_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    names_yaml = "\n".join(f"  {idx}: {_yaml_name(name)}" for idx, name in enumerate(class_names))
    data_yaml_content = f"""path: {run_output_dir.absolute()}
train: images/train
val: images/val
test: images/test
nc: {len(class_names)}
names:
{names_yaml}
"""
    (run_output_dir / "data.yaml").write_text(data_yaml_content, encoding="utf-8")
    print(f"[INFO] data.yaml generated at {run_output_dir / 'data.yaml'}")

    print(f"\n[SUCCESS] Mixed dataset generated at {run_output_dir}")
    print(f"  train (original): {train_orig_count}")
    print(f"  train (generated): {generated_count}")
    print(f"  train (total): {train_orig_count + generated_count}")
    print(f"  val: {mixed_manifest['val_count']}")
    print(f"  test: {mixed_manifest['test_count']}")


def find_latest_run_dir(generated_dir):
    generated_path = Path(generated_dir)
    if not generated_path.exists():
        return None
    run_dirs = [d for d in generated_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a mixed dataset from original and generated defect images.")
    parser.add_argument("--orig_images_dir", default=r"D:\VScode\lora\NEU-DET\IMAGES", help="Original image root")
    parser.add_argument("--orig_labels_dir", default=r"D:\VScode\lora\outputs\yolo_baseline\labels", help="Original YOLO label root")
    parser.add_argument("--manifest_path", default=r"D:\VScode\lora\outputs\split_manifest.json", help="split_manifest.json path")
    parser.add_argument("--new_images_dir", default=None, help="Directory containing generated images or an images subdir")
    parser.add_argument("--generated_base_dir", default=r"D:\VScode\lora\outputs\generated", help="Base directory used for auto-discovery")
    parser.add_argument("--run_output_dir", default="", help="Output directory for mixed dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--score_path", default=None, help="candidate_scores.jsonl used for quality-aware selection")
    parser.add_argument("--max_generated", type=int, default=None, help="Maximum generated images copied into train")
    parser.add_argument("--per_source_limit", type=int, default=0, help="Maximum generated images per original source image; 0 keeps all")
    parser.add_argument("--per_class_limit", type=int, default=0, help="Maximum generated images per generated class; 0 keeps all")
    parser.add_argument("--quality_threshold", type=float, default=None, help="Minimum generator selection score for generated inclusion")
    parser.add_argument("--min_canvas_drop", type=float, default=None, help="Minimum canvas.quality.defect_evidence_drop")
    parser.add_argument("--max_outside_delta", type=float, default=None, help="Maximum quality.outside_delta")
    parser.add_argument(
        "--selection_strategy",
        default="random",
        choices=["random", "quality", "balanced-quality", "steel-utility", "balanced-steel-utility"],
        help="Generated-image selection strategy",
    )
    parser.add_argument("--auto", action="store_true", help="Automatically pick the latest generated run")
    args = parser.parse_args()

    if args.auto or args.new_images_dir is None:
        latest_run = find_latest_run_dir(args.generated_base_dir)
        if latest_run is None:
            print(f"[ERROR] Could not find generated run under {args.generated_base_dir}")
            raise SystemExit(1)
        new_images_dir = latest_run
        print(f"[INFO] Auto-detected latest generation directory: {new_images_dir}")
    else:
        new_images_dir = args.new_images_dir

    create_mixed_dataset(
        args.orig_images_dir,
        args.orig_labels_dir,
        args.manifest_path,
        new_images_dir,
        args.run_output_dir,
        seed=args.seed,
        score_path=args.score_path,
        max_generated=args.max_generated,
        per_source_limit=args.per_source_limit,
        per_class_limit=args.per_class_limit,
        quality_threshold=args.quality_threshold,
        min_canvas_drop=args.min_canvas_drop,
        max_outside_delta=args.max_outside_delta,
        selection_strategy=args.selection_strategy,
    )
