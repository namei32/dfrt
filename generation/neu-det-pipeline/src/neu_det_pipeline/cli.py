from __future__ import annotations

import csv
import json
import importlib.util
import logging
import sys
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Any, List

import typer
import torch
import shutil
from rich.console import Console

from .config import (
    ConfigBundle,
    DatasetConfig,
    GenerationConfig,
    GuidanceConfig,
    LoRAConfig,
    load_config_bundle,
)
# Data module
from .data import collect_dataset, collect_dataset_images, collect_dataset_instances, split_dataset, compute_class_counts, create_mixed_dataset
# Guidance module
from .guidance import GuidanceExtractor
# Prompts module
from .prompts import CaptionGenerator, load_captions_from_file, generate_captions_with_blip2
# Models module
from .models import (
    ContextSGDAConfig,
    ContextSGDAGenerator,
    ContextLGIConfig,
    ContextLGIGenerator,
    DELGIConfig,
    DELGIGenerator,
    DBTCMDPGenerator,
    DRFTGenerator,
    DRFTLoRAHyperParams,
    DRFTLoRATrainer,
    ReferenceCMDPGenerator,
    ReferenceLDMTrainer,
)
from .models.drft_lora import is_drft_lora_path, read_drft_lora_metadata
from .models.reference_cmdp import infer_surface_domain, slugify_class_name
# Evaluation module
from .evaluation import GenerationMetricsEvaluator, save_metrics

app = typer.Typer(add_completion=False)
console = Console()
DEFAULT_BUNDLE = load_config_bundle()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_split_manifest() -> Path:
    return _project_root() / "data" / "processed" / "split_manifest.json"


def _load_split_stems(manifest_path: Path, split: str) -> set[str]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    stems = manifest.get(split)
    if not isinstance(stems, list):
        raise ValueError(f"Split '{split}' not found in {manifest_path}")
    return {str(stem) for stem in stems}


def _filter_samples_by_stems(samples: List[Any], stems: set[str]) -> List[Any]:
    return [sample for sample in samples if sample.image_path.stem in stems]


def _filter_samples_for_split(
    samples: List[Any],
    split: str,
    manifest_path: Optional[Path] = None,
) -> List[Any]:
    split_lc = split.lower().strip()
    if not split_lc or split_lc == "all":
        return samples
    if manifest_path is not None and Path(manifest_path).exists():
        stems = _load_split_stems(Path(manifest_path), split_lc)
        return _filter_samples_by_stems(samples, stems)
    split_samples = []
    for sample in samples:
        ann_path = getattr(sample, "annotation_path", None)
        if ann_path is not None and Path(ann_path).parent.name.lower() == split_lc:
            split_samples.append(sample)
    return split_samples if split_samples else samples


def _limit_samples_by_strategy(
    samples: List[Any],
    max_samples: Optional[int],
    *,
    strategy: str = "input-order",
) -> List[Any]:
    if max_samples is None:
        return samples
    limit = max(1, int(max_samples))
    strategy_key = strategy.lower().strip().replace("_", "-")
    if strategy_key not in {"balanced", "round-robin"}:
        return samples[:limit]

    groups: Dict[str, List[Any]] = {}
    for sample in samples:
        groups.setdefault(str(sample.cls_name), []).append(sample)
    selected: List[Any] = []
    class_names = sorted(groups)
    cursor = 0
    used_sources: set[str] = set()
    while len(selected) < limit and any(groups.values()):
        cls_name = class_names[cursor % len(class_names)]
        if groups[cls_name]:
            group = groups[cls_name]
            pick_idx = 0
            for idx, candidate in enumerate(group):
                source_key = str(getattr(candidate, "source_stem", candidate.image_path.stem))
                if source_key not in used_sources:
                    pick_idx = idx
                    break
            sample = group.pop(pick_idx)
            used_sources.add(str(getattr(sample, "source_stem", sample.image_path.stem)))
            selected.append(sample)
        cursor += 1
    return selected


def _write_split_manifest_from_voc_layout(dataset_root: Path, out_path: Path) -> Path:
    ann_root = dataset_root / "annotations"
    manifest: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for split in list(manifest):
        split_dir = ann_root / split
        if split_dir.exists():
            manifest[split] = sorted(path.stem for path in split_dir.glob("*.xml"))
    if not any(manifest.values()):
        all_stems = sorted(path.stem for path in (dataset_root / "ANNOTATIONS").glob("*.xml"))
        manifest["train"] = all_stems
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _default_images_dir_for_dataset(dataset_root: Path) -> Path:
    if (dataset_root / "IMAGES").exists():
        return dataset_root / "IMAGES"
    if (dataset_root / "images").exists():
        return dataset_root / "images"
    return dataset_root


def _default_labels_dir_for_dataset(dataset_root: Path) -> Path:
    if (dataset_root / "labels").exists():
        return dataset_root / "labels"
    return _project_root() / "data" / "processed" / "yolo" / "labels"


def _read_yolo_class_names(dataset_root: Path) -> Dict[str, int]:
    data_yaml = Path(dataset_root) / "data.yaml"
    names: Dict[str, int] = {}
    if not data_yaml.exists():
        return names
    for raw in data_yaml.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if ":" not in line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key.isdigit() and value:
            names[value] = int(key)
    return names


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


_HDSI_S2C_CSV_FIELDS = [
    "class",
    "target_key",
    "source_key",
    "candidate_index",
    "selected",
    "candidate_accepted",
    "selection_score",
    "output_path",
    "candidate_path",
    "rejection_reasons",
    "hdsi_hardness_score",
    "hdsi_validity_score",
    "hdsi_source_distance",
    "hdsi_intervention_score",
    "hdsi_validity_gate_pass",
    "hdsi_source_distance_gate_pass",
    "hdsi_hard_valid_gate_pass",
    "hdsi_s2c_score",
    "hdsi_s2c_structure_score",
    "hdsi_s2c_image_score",
    "hdsi_s2c_spectrum_match",
    "hdsi_s2c_spectral_distance",
    "hdsi_s2c_selection_score",
    "hdsi_s2c_gate_pass",
    "hdsi_s2c_gate_failures",
    "hdsi_s2c_measurement_stage",
    "hdsi_s2c_area_score",
    "hdsi_s2c_orientation_score",
    "hdsi_s2c_roughness_score",
    "hdsi_s2c_density_score",
    "hdsi_s2c_contrast_score",
    "hdsi_s2c_visible_score",
    "inside_delta",
    "outside_delta",
    "source_similarity",
    "s2i_score",
    "s2i_ratio",
    "visible_change_fraction",
    "generated_low_freq_ratio",
    "generated_mid_freq_ratio",
    "generated_high_freq_ratio",
    "generated_area_fraction",
    "generated_component_density",
    "generated_boundary_roughness",
    "generated_elongation",
    "generated_contrast_ratio",
]


def _nested_value(source: Any, path: str) -> Any:
    current = source
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _hdsi_s2c_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    hdsi = _nested_value(row, "utility_causal_spectrum.hdsi") or {}
    hdsi_s2c = _nested_value(row, "utility_causal_spectrum.hdsi_s2c") or {}
    generated = quality.get("hdsi_s2c_generated_spectrum")
    if not isinstance(generated, dict):
        generated = hdsi_s2c.get("hdsi_s2c_generated_spectrum") if isinstance(hdsi_s2c, dict) else {}
    if not isinstance(generated, dict):
        generated = {}
    paths = row.get("paths") if isinstance(row.get("paths"), dict) else {}
    flat = {
        "class": row.get("class"),
        "target_key": row.get("target_key"),
        "source_key": row.get("source_key"),
        "candidate_index": row.get("candidate_index"),
        "selected": row.get("selected"),
        "candidate_accepted": row.get("candidate_accepted"),
        "selection_score": row.get("selection_score"),
        "output_path": row.get("output_path"),
        "candidate_path": row.get("candidate_path") or paths.get("candidate"),
        "rejection_reasons": row.get("rejection_reasons") or [],
        "hdsi_hardness_score": hdsi.get("hardness_score"),
        "hdsi_validity_score": hdsi.get("validity_score") or quality.get("hdsi_validity_score"),
        "hdsi_source_distance": hdsi.get("source_distance"),
        "hdsi_intervention_score": hdsi.get("intervention_score") or quality.get("hdsi_intervention_score"),
        "hdsi_validity_gate_pass": hdsi.get("validity_gate_pass"),
        "hdsi_source_distance_gate_pass": hdsi.get("source_distance_gate_pass"),
        "hdsi_hard_valid_gate_pass": hdsi.get("hard_valid_gate_pass"),
        "hdsi_s2c_score": quality.get("hdsi_s2c_score") or hdsi_s2c.get("hdsi_s2c_score"),
        "hdsi_s2c_structure_score": quality.get("hdsi_s2c_structure_score") or hdsi_s2c.get("hdsi_s2c_structure_score"),
        "hdsi_s2c_image_score": quality.get("hdsi_s2c_image_score") or hdsi_s2c.get("hdsi_s2c_image_score"),
        "hdsi_s2c_spectrum_match": quality.get("hdsi_s2c_spectrum_match") or hdsi_s2c.get("hdsi_s2c_spectrum_match"),
        "hdsi_s2c_spectral_distance": quality.get("hdsi_s2c_spectral_distance") or hdsi_s2c.get("hdsi_s2c_spectral_distance"),
        "hdsi_s2c_selection_score": quality.get("hdsi_s2c_selection_score"),
        "hdsi_s2c_gate_pass": quality.get("hdsi_s2c_gate_pass") if "hdsi_s2c_gate_pass" in quality else hdsi_s2c.get("hdsi_s2c_gate_pass"),
        "hdsi_s2c_gate_failures": quality.get("hdsi_s2c_gate_failures") or hdsi_s2c.get("hdsi_s2c_gate_failures") or [],
        "hdsi_s2c_measurement_stage": quality.get("hdsi_s2c_measurement_stage") or hdsi_s2c.get("hdsi_s2c_measurement_stage"),
        "hdsi_s2c_area_score": quality.get("hdsi_s2c_area_score") or hdsi_s2c.get("hdsi_s2c_area_score"),
        "hdsi_s2c_orientation_score": quality.get("hdsi_s2c_orientation_score") or hdsi_s2c.get("hdsi_s2c_orientation_score"),
        "hdsi_s2c_roughness_score": quality.get("hdsi_s2c_roughness_score") or hdsi_s2c.get("hdsi_s2c_roughness_score"),
        "hdsi_s2c_density_score": quality.get("hdsi_s2c_density_score") or hdsi_s2c.get("hdsi_s2c_density_score"),
        "hdsi_s2c_contrast_score": quality.get("hdsi_s2c_contrast_score") or hdsi_s2c.get("hdsi_s2c_contrast_score"),
        "hdsi_s2c_visible_score": quality.get("hdsi_s2c_visible_score") or hdsi_s2c.get("hdsi_s2c_visible_score"),
        "inside_delta": quality.get("inside_delta"),
        "outside_delta": quality.get("outside_delta"),
        "source_similarity": quality.get("source_similarity"),
        "s2i_score": quality.get("s2i_score"),
        "s2i_ratio": quality.get("s2i_ratio"),
        "visible_change_fraction": quality.get("visible_change_fraction"),
        "generated_low_freq_ratio": generated.get("low_freq_ratio"),
        "generated_mid_freq_ratio": generated.get("mid_freq_ratio"),
        "generated_high_freq_ratio": generated.get("high_freq_ratio"),
        "generated_area_fraction": generated.get("area_fraction"),
        "generated_component_density": generated.get("component_density"),
        "generated_boundary_roughness": generated.get("boundary_roughness"),
        "generated_elongation": generated.get("elongation"),
        "generated_contrast_ratio": generated.get("contrast_ratio"),
    }
    return {key: _csv_value(flat.get(key)) for key in _HDSI_S2C_CSV_FIELDS}


def _write_hdsi_s2c_csv_exports(run_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    paths = {
        "candidate_scores_csv": Path(run_dir) / "hdsi_s2c_candidate_scores.csv",
        "selected_scores_csv": Path(run_dir) / "hdsi_s2c_selected_scores.csv",
    }
    for path, group in (
        (paths["candidate_scores_csv"], rows),
        (paths["selected_scores_csv"], [row for row in rows if bool(row.get("selected", False))]),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=_HDSI_S2C_CSV_FIELDS)
            writer.writeheader()
            for row in group:
                writer.writerow(_hdsi_s2c_csv_row(row))
    return {key: str(path.resolve()) for key, path in paths.items()}


def _build_spec_srt_summary(run_dir: Path, final_manifest: Dict[str, Any]) -> Dict[str, Any]:
    score_path = Path(str(final_manifest.get("score_path", "")))
    rows = _load_jsonl(score_path)
    by_class: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "candidate_rows": 0,
            "attempted_targets": set(),
            "accepted_candidates": 0,
            "accepted_targets": set(),
            "selected_images": 0,
            "selected_targets": set(),
            "gate_failures": Counter(),
            "rejection_reasons": Counter(),
            "selected_quality": [],
            "thresholds": {},
        }
    )
    global_failures: Counter[str] = Counter()
    global_reasons: Counter[str] = Counter()
    for row in rows:
        cls_name = str(row.get("class", "unknown"))
        target_key = str(row.get("target_key", ""))
        item = by_class[cls_name]
        item["candidate_rows"] += 1
        if target_key:
            item["attempted_targets"].add(target_key)
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        for tag in quality.get("spec_srt_gate_failures", []) if isinstance(quality, dict) else []:
            item["gate_failures"][str(tag)] += 1
            global_failures[str(tag)] += 1
        for reason in row.get("rejection_reasons", []) or []:
            item["rejection_reasons"][str(reason)] += 1
            global_reasons[str(reason)] += 1
        if isinstance(quality, dict) and isinstance(quality.get("spec_srt_gate_thresholds"), dict):
            item["thresholds"] = dict(quality["spec_srt_gate_thresholds"])
        if bool(row.get("candidate_accepted", False)):
            item["accepted_candidates"] += 1
            if target_key:
                item["accepted_targets"].add(target_key)
        if bool(row.get("selected", False)):
            item["selected_images"] += 1
            if target_key:
                item["selected_targets"].add(target_key)
            if isinstance(quality, dict):
                item["selected_quality"].append({
                    "selection_score": float(row.get("selection_score", 0.0) or 0.0),
                    "inside_delta": float(quality.get("inside_delta", 0.0) or 0.0),
                    "visible_change_fraction": float(quality.get("visible_change_fraction", 0.0) or 0.0),
                    "source_similarity": float(quality.get("source_similarity", 0.0) or 0.0),
                    "sharpness_ratio": float(quality.get("sharpness_ratio", 0.0) or 0.0),
                    "s2i_ratio": float(quality.get("s2i_ratio", 0.0) or 0.0),
                    "source_spectrum_distance": float(quality.get("source_spectrum_distance", 0.0) or 0.0),
                })

    def _mean_metric(items: List[Dict[str, float]], key: str) -> float | None:
        values = [float(item[key]) for item in items if key in item]
        return float(sum(values) / len(values)) if values else None

    classes: Dict[str, Any] = {}
    for cls_name, item in sorted(by_class.items()):
        selected_quality = list(item["selected_quality"])
        classes[cls_name] = {
            "candidate_rows": int(item["candidate_rows"]),
            "attempted_targets": len(item["attempted_targets"]),
            "accepted_candidates": int(item["accepted_candidates"]),
            "accepted_targets": len(item["accepted_targets"]),
            "selected_images": int(item["selected_images"]),
            "skipped_targets": max(0, len(item["attempted_targets"]) - len(item["selected_targets"])),
            "gate_failures": dict(sorted(item["gate_failures"].items())),
            "rejection_reasons": dict(sorted(item["rejection_reasons"].items())),
            "thresholds": item["thresholds"],
            "selected_quality_mean": {
                key: _mean_metric(selected_quality, key)
                for key in (
                    "selection_score",
                    "inside_delta",
                    "visible_change_fraction",
                    "source_similarity",
                    "sharpness_ratio",
                    "s2i_ratio",
                    "source_spectrum_distance",
                )
            },
        }
    summary = {
        "method": final_manifest.get("method"),
        "preset": final_manifest.get("preset"),
        "run_dir": str(Path(run_dir).resolve()),
        "score_path": str(score_path.resolve()) if score_path.exists() else str(score_path),
        "candidate_rows": len(rows),
        "generated_images": int(final_manifest.get("generated_images", 0) or 0),
        "selected_rows": sum(1 for row in rows if bool(row.get("selected", False))),
        "accepted_candidates": sum(1 for row in rows if bool(row.get("candidate_accepted", False))),
        "global_gate_failures": dict(sorted(global_failures.items())),
        "global_rejection_reasons": dict(sorted(global_reasons.items())),
        "classes": classes,
        "disabled_legacy_modules": {
            "drr": not bool(final_manifest.get("config", {}).get("use_drr", False)),
            "band_recomposition": not bool(final_manifest.get("config", {}).get("use_band_recomposition", False)),
            "hdsi_pd": not bool(final_manifest.get("config", {}).get("use_hdsi_pd", False)),
            "seca": not bool(final_manifest.get("config", {}).get("use_seca", False)),
            "bbox_update": not bool(final_manifest.get("config", {}).get("srt_bbox_update", False)),
        },
    }
    summary_path = Path(run_dir) / "spec_srt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _select_balanced_generated_rows(rows: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    selected = [
        row
        for row in rows
        if bool(row.get("selected", False)) and row.get("output_path") and Path(str(row["output_path"])).exists()
    ]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[str(row.get("class", "unknown"))].append(row)
    for cls_rows in groups.values():
        cls_rows.sort(key=lambda row: float(row.get("selection_score", 0.0) or 0.0), reverse=True)
    chosen: List[Dict[str, Any]] = []
    while len(chosen) < target_count and any(groups.values()):
        ordered = sorted(groups, key=lambda cls: (sum(1 for row in chosen if str(row.get("class", "unknown")) == cls), cls))
        progressed = False
        for cls in ordered:
            if not groups[cls]:
                continue
            chosen.append(groups[cls].pop(0))
            progressed = True
            if len(chosen) >= target_count:
                break
        if not progressed:
            break
    return chosen


def _write_curated_generated_dataset(
    run_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    dataset_root: Path,
    target_count: int,
    curated_slug: str = "spec_srt",
    selection_label: str = "balanced-by-class-then-selection-score",
) -> Dict[str, Any]:
    safe_slug = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in curated_slug).strip("_")
    curated_dir = Path(run_dir) / f"curated_{safe_slug or 'generated'}_{target_count}"
    image_dir = curated_dir / "images"
    label_dir = curated_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = _read_yolo_class_names(dataset_root)
    if not class_to_id:
        class_names = sorted({str(row.get("class", "unknown")) for row in rows})
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
    copied_rows: List[Dict[str, Any]] = []
    for row in rows:
        src = Path(str(row["output_path"]))
        dst = image_dir / src.name
        shutil.copy2(src, dst)
        spatial = row.get("spatial_code") if isinstance(row.get("spatial_code"), dict) else {}
        bbox = spatial.get("generated_bbox") if isinstance(spatial, dict) else None
        cls_name = str(row.get("class", "unknown"))
        class_id = int(class_to_id.get(cls_name, 0))
        if isinstance(bbox, list) and len(bbox) == 4:
            from PIL import Image

            with Image.open(dst) as image:
                width, height = image.size
            x1, y1, x2, y2 = [float(v) for v in bbox]
            xc = ((x1 + x2) / 2.0) / max(1.0, float(width))
            yc = ((y1 + y2) / 2.0) / max(1.0, float(height))
            bw = max(0.0, x2 - x1) / max(1.0, float(width))
            bh = max(0.0, y2 - y1) / max(1.0, float(height))
            label_dir.joinpath(dst.with_suffix(".txt").name).write_text(
                f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n",
                encoding="utf-8",
            )
        copied = dict(row)
        copied["curated_output_path"] = str(dst.resolve())
        copied_rows.append(copied)
    names_lines = "\n".join(f"  {idx}: {name}" for name, idx in sorted(class_to_id.items(), key=lambda item: item[1]))
    (curated_dir / "data.yaml").write_text(
        f"path: .\ntrain: images\nval: images\nnc: {len(class_to_id)}\nnames:\n{names_lines}\n",
        encoding="utf-8",
    )
    _write_jsonl(curated_dir / "selected_scores.jsonl", copied_rows)
    manifest = {
        "target_count": int(target_count),
        "selected_images": len(copied_rows),
        "shortfall": max(0, int(target_count) - len(copied_rows)),
        "curated_dir": str(curated_dir.resolve()),
        "image_dir": str(image_dir.resolve()),
        "label_dir": str(label_dir.resolve()),
        "class_counts": dict(sorted(Counter(str(row.get("class", "unknown")) for row in copied_rows).items())),
        "selection": selection_label,
        "source_method": safe_slug or "generated",
    }
    (curated_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _mean_numeric(rows: List[Dict[str, Any]], key: str, *, nested: str | None = None) -> float | None:
    values: List[float] = []
    for row in rows:
        source: Any = row
        if nested:
            for part in nested.split("."):
                source = source.get(part) if isinstance(source, dict) else None
        if isinstance(source, dict):
            value = source.get(key)
        else:
            value = None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return float(sum(values) / len(values)) if values else None


def _build_e_srt_hdsi_pd_summary(run_dir: Path, final_manifest: Dict[str, Any]) -> Dict[str, Any]:
    score_path = Path(str(final_manifest.get("score_path", "")))
    rows = _load_jsonl(score_path)
    selected = [row for row in rows if bool(row.get("selected", False))]
    accepted = [row for row in rows if bool(row.get("candidate_accepted", False))]

    def _pd_enabled(row: Dict[str, Any]) -> bool:
        srt = row.get("structure_residual_transport")
        pd = srt.get("hdsi_phase_detail_intervention") if isinstance(srt, dict) else None
        return bool(pd.get("hdsi_pd_enabled", False)) if isinstance(pd, dict) else False

    def _metric_means(group: List[Dict[str, Any]]) -> Dict[str, float | None]:
        return {
            "selection_score": _mean_numeric(group, "selection_score"),
            "inside_delta": _mean_numeric(group, "inside_delta", nested="quality"),
            "outside_delta": _mean_numeric(group, "outside_delta", nested="quality"),
            "s2i_score": _mean_numeric(group, "s2i_score", nested="quality"),
            "s2i_visible_delta": _mean_numeric(group, "s2i_visible_delta", nested="quality"),
            "s2i_ratio": _mean_numeric(group, "s2i_ratio", nested="quality"),
            "hdsi_intervention_score": _mean_numeric(group, "hdsi_intervention_score", nested="quality"),
            "hdsi_validity_score": _mean_numeric(group, "hdsi_validity_score", nested="quality"),
            "hdsi_pd_selection_score": _mean_numeric(group, "hdsi_pd_selection_score", nested="quality"),
            "hdsi_pd_detail_score": _mean_numeric(group, "hdsi_pd_detail_score", nested="quality"),
            "e_srt_hdsi_pd_selection_score": _mean_numeric(group, "e_srt_hdsi_pd_selection_score", nested="quality"),
            "e_srt_hdsi_pd_useful_change_score": _mean_numeric(group, "e_srt_hdsi_pd_useful_change_score", nested="quality"),
            "e_srt_hdsi_pd_copy_escape_score": _mean_numeric(group, "e_srt_hdsi_pd_copy_escape_score", nested="quality"),
            "e_srt_hdsi_pd_focus_score": _mean_numeric(group, "e_srt_hdsi_pd_focus_score", nested="quality"),
        }

    by_class: Dict[str, Dict[str, Any]] = {}
    for cls_name in sorted({str(row.get("class", "unknown")) for row in rows}):
        cls_rows = [row for row in rows if str(row.get("class", "unknown")) == cls_name]
        cls_selected = [row for row in cls_rows if bool(row.get("selected", False))]
        cls_accepted = [row for row in cls_rows if bool(row.get("candidate_accepted", False))]
        by_class[cls_name] = {
            "candidate_rows": len(cls_rows),
            "accepted_candidates": len(cls_accepted),
            "selected_images": len(cls_selected),
            "pd_enabled_rows": sum(1 for row in cls_rows if _pd_enabled(row)),
            "gate_pass_rows": sum(
                1
                for row in cls_rows
                if bool((row.get("quality") if isinstance(row.get("quality"), dict) else {}).get("e_srt_hdsi_pd_gate_pass", False))
            ),
            "rejection_reasons": dict(
                sorted(Counter(str(reason) for row in cls_rows for reason in (row.get("rejection_reasons") or [])).items())
            ),
            "selected_quality_mean": _metric_means(cls_selected),
        }

    summary = {
        "method": final_manifest.get("method"),
        "preset": final_manifest.get("preset"),
        "run_dir": str(Path(run_dir).resolve()),
        "score_path": str(score_path.resolve()) if score_path.exists() else str(score_path),
        "candidate_rows": len(rows),
        "accepted_candidates": len(accepted),
        "selected_rows": len(selected),
        "generated_images": int(final_manifest.get("generated_images", 0) or 0),
        "pd_enabled_rows": sum(1 for row in rows if _pd_enabled(row)),
        "gate_pass_rows": sum(
            1
            for row in rows
            if bool((row.get("quality") if isinstance(row.get("quality"), dict) else {}).get("e_srt_hdsi_pd_gate_pass", False))
        ),
        "global_rejection_reasons": dict(
            sorted(Counter(str(reason) for row in rows for reason in (row.get("rejection_reasons") or [])).items())
        ),
        "selected_quality_mean": _metric_means(selected),
        "accepted_quality_mean": _metric_means(accepted),
        "classes": by_class,
        "disabled_legacy_modules": {
            "drr": not bool(final_manifest.get("config", {}).get("use_drr", False)),
            "band_recomposition": not bool(final_manifest.get("config", {}).get("use_band_recomposition", False)),
            "seca": not bool(final_manifest.get("config", {}).get("use_seca", False)),
            "bbox_update": not bool(final_manifest.get("config", {}).get("srt_bbox_update", False)),
        },
    }
    summary_path = Path(run_dir) / "e_srt_hdsi_pd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _build_hdsi_s2c_summary(run_dir: Path, final_manifest: Dict[str, Any]) -> Dict[str, Any]:
    score_path = Path(str(final_manifest.get("score_path", "")))
    rows = _load_jsonl(score_path)
    selected = [row for row in rows if bool(row.get("selected", False))]
    accepted = [row for row in rows if bool(row.get("candidate_accepted", False))]
    csv_paths = _write_hdsi_s2c_csv_exports(Path(run_dir), rows)

    def _metric_means(group: List[Dict[str, Any]]) -> Dict[str, float | None]:
        return {
            "selection_score": _mean_numeric(group, "selection_score"),
            "inside_delta": _mean_numeric(group, "inside_delta", nested="quality"),
            "outside_delta": _mean_numeric(group, "outside_delta", nested="quality"),
            "source_similarity": _mean_numeric(group, "source_similarity", nested="quality"),
            "s2i_score": _mean_numeric(group, "s2i_score", nested="quality"),
            "s2i_ratio": _mean_numeric(group, "s2i_ratio", nested="quality"),
            "hdsi_intervention_score": _mean_numeric(group, "hdsi_intervention_score", nested="quality"),
            "hdsi_validity_score": _mean_numeric(group, "hdsi_validity_score", nested="quality"),
            "hdsi_s2c_score": _mean_numeric(group, "hdsi_s2c_score", nested="quality"),
            "hdsi_s2c_structure_score": _mean_numeric(group, "hdsi_s2c_structure_score", nested="quality"),
            "hdsi_s2c_image_score": _mean_numeric(group, "hdsi_s2c_image_score", nested="quality"),
            "hdsi_s2c_spectrum_match": _mean_numeric(group, "hdsi_s2c_spectrum_match", nested="quality"),
            "hdsi_s2c_selection_score": _mean_numeric(group, "hdsi_s2c_selection_score", nested="quality"),
            "source_spectrum_distance": _mean_numeric(group, "source_spectrum_distance", nested="quality"),
        }

    by_class: Dict[str, Dict[str, Any]] = {}
    for cls_name in sorted({str(row.get("class", "unknown")) for row in rows}):
        cls_rows = [row for row in rows if str(row.get("class", "unknown")) == cls_name]
        cls_selected = [row for row in cls_rows if bool(row.get("selected", False))]
        cls_accepted = [row for row in cls_rows if bool(row.get("candidate_accepted", False))]
        by_class[cls_name] = {
            "candidate_rows": len(cls_rows),
            "accepted_candidates": len(cls_accepted),
            "selected_images": len(cls_selected),
            "gate_pass_rows": sum(
                1
                for row in cls_rows
                if bool((row.get("quality") if isinstance(row.get("quality"), dict) else {}).get("hdsi_s2c_gate_pass", False))
            ),
            "gate_failures": dict(
                sorted(
                    Counter(
                        str(reason)
                        for row in cls_rows
                        for reason in (
                            (row.get("quality") if isinstance(row.get("quality"), dict) else {}).get("hdsi_s2c_gate_failures", [])
                            or []
                        )
                    ).items()
                )
            ),
            "rejection_reasons": dict(
                sorted(Counter(str(reason) for row in cls_rows for reason in (row.get("rejection_reasons") or [])).items())
            ),
            "selected_quality_mean": _metric_means(cls_selected),
        }

    summary = {
        "method": final_manifest.get("method"),
        "preset": final_manifest.get("preset"),
        "run_dir": str(Path(run_dir).resolve()),
        "score_path": str(score_path.resolve()) if score_path.exists() else str(score_path),
        "csv_paths": csv_paths,
        "candidate_rows": len(rows),
        "accepted_candidates": len(accepted),
        "selected_rows": len(selected),
        "generated_images": int(final_manifest.get("generated_images", 0) or 0),
        "gate_pass_rows": sum(
            1
            for row in rows
            if bool((row.get("quality") if isinstance(row.get("quality"), dict) else {}).get("hdsi_s2c_gate_pass", False))
        ),
        "global_gate_failures": dict(
            sorted(
                Counter(
                    str(reason)
                    for row in rows
                    for reason in (
                        (row.get("quality") if isinstance(row.get("quality"), dict) else {}).get("hdsi_s2c_gate_failures", [])
                        or []
                    )
                ).items()
            )
        ),
        "global_rejection_reasons": dict(
            sorted(Counter(str(reason) for row in rows for reason in (row.get("rejection_reasons") or [])).items())
        ),
        "selected_quality_mean": _metric_means(selected),
        "accepted_quality_mean": _metric_means(accepted),
        "classes": by_class,
        "disabled_legacy_modules": {
            "drr": not bool(final_manifest.get("config", {}).get("use_drr", False)),
            "band_recomposition": not bool(final_manifest.get("config", {}).get("use_band_recomposition", False)),
            "hdsi_pd": not bool(final_manifest.get("config", {}).get("use_hdsi_pd", False)),
            "seca": not bool(final_manifest.get("config", {}).get("use_seca", False)),
            "bbox_update": not bool(final_manifest.get("config", {}).get("srt_bbox_update", False)),
        },
    }
    summary_path = Path(run_dir) / "hdsi_s2c_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _sample_target_key(sample: Any) -> str:
    return str(getattr(sample, "target_key", sample.image_path.stem))


def _find_workspace_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "local_ultralytics.py").exists() and (parent / "ultralytics" / "ultralytics").exists():
            return parent
    return None


def _ensure_local_yolo() -> Path:
    workspace_root = _find_workspace_root()
    if workspace_root is None:
        raise RuntimeError("Could not find the bundled workspace Ultralytics fork.")
    sys.path.insert(0, str(workspace_root))
    from local_ultralytics import ensure_local_ultralytics

    return ensure_local_ultralytics()


def _ensure_bundle(ctx: typer.Context) -> ConfigBundle:
    bundle = ctx.obj
    if isinstance(bundle, ConfigBundle):
        return bundle
    return DEFAULT_BUNDLE


def _load_local_run_all_module():
    workspace_root = _ensure_local_yolo()
    run_all_path = workspace_root / "experiments" / "run_all.py"
    if not run_all_path.exists():
        raise FileNotFoundError(f"Could not find local run_all.py at {run_all_path}")
    module_name = "_local_yolov8_run_all"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, run_all_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {run_all_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_yolo_backend(train_mode: str, model: str, data_yaml: Path):
    normalized = (train_mode or "default").strip() or "default"
    if normalized == "default":
        _ensure_local_yolo()
        from ultralytics import YOLO

        return YOLO, None

    run_all = _load_local_run_all_module()
    args = SimpleNamespace(disable_balanced_mixed_sampler=False)
    exp = run_all.Experiment(name="adhoc", model=model, data=str(data_yaml.resolve()), train_mode=normalized)
    trainer_cls = run_all.make_trainer_cls(exp, args)
    return run_all.YOLO, trainer_cls


def _load_lora_metadata(lora_path: Path, fallback_cfg: LoRAConfig) -> Dict[str, Any]:
    cfg_file = lora_path.parent / "lora_config.json"
    if cfg_file.exists():
        try:
            return json.loads(cfg_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            console.print(f"[yellow]无法解析 {cfg_file}，将使用默认 LoRA 配置。[/yellow]")
    return {
        "model_id": fallback_cfg.model_id,
        "rank": fallback_cfg.rank,
        "alpha": fallback_cfg.alpha,
        "dropout_rate": getattr(fallback_cfg, "dropout_rate", 0.0),
        "target_modules": [],
        "resolution": fallback_cfg.resolution,
        "seed": fallback_cfg.seed,
        "prompt_template": fallback_cfg.prompt_template,
        "mixed_precision": fallback_cfg.mixed_precision,
        "training_hyperparameters": {
            "learning_rate": fallback_cfg.learning_rate,
            "steps": fallback_cfg.steps,
            "batch_size": fallback_cfg.batch_size,
            "gradient_accumulation_steps": fallback_cfg.gradient_accumulation_steps,
            "optimizer": "AdamW",
            "lr_scheduler": fallback_cfg.lr_scheduler,
            "lr_warmup_steps": fallback_cfg.lr_warmup_steps,
            "max_grad_norm": fallback_cfg.max_grad_norm,
        },
    }



def _setup_logging(log_file: Optional[Path]) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = log_file.expanduser().resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_file, encoding="utf-8"))
        console.print(f"日志将写入 {log_file}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="YAML 配置文件路径（覆盖默认 config.yaml）",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
) -> None:
    """NEU-DET data augmentation pipeline."""

    if config is not None:
        try:
            ctx.obj = load_config_bundle(config)
        except FileNotFoundError as exc:  # pragma: no cover - CLI validation
            raise typer.BadParameter(str(exc)) from exc
    else:
        ctx.obj = DEFAULT_BUNDLE


@app.command()
def prepare(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    test_size: Optional[float] = typer.Option(None, help="验证集比例 (默认取配置文件中的值)"),
) -> None:
    bundle = _ensure_bundle(ctx)
    dataset_cfg = replace(bundle.dataset, root=dataset_root)
    effective_test_size = test_size if test_size is not None else dataset_cfg.test_size
    samples = collect_dataset(dataset_cfg.root)
    splits = split_dataset(samples, test_size=effective_test_size, seed=dataset_cfg.seed)
    counts = compute_class_counts(samples)
    console.print("Loaded", len(samples), "samples")
    console.print(counts)



@app.command()
def guidance(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/guidance"), file_okay=False),
) -> None:
    bundle = _ensure_bundle(ctx)
    cfg = replace(bundle.guidance)
    samples = collect_dataset(dataset_root)
    extractor = GuidanceExtractor(cfg.hed_repo_id, cfg.midas_repo_id, cfg.hed_ckpt, cfg.midas_ckpt)
    outputs = extractor.batch_process(samples, output_dir)
    console.print("Stored guidance cues for", len(outputs), "samples")


@app.command()
def caption(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_file: Path = typer.Option(Path("outputs/captions.json"), file_okay=True),
    model_name: str = typer.Option("openai/clip-vit-large-patch14", help="CLIP model name (ignored if --use-blip2 is set)"),
    use_paper_keywords: bool = typer.Option(True, help="Use paper-specified keywords (paper format) or CLIP-selected templates"),
    lora_weight: float = typer.Option(1.0, help="LoRA weight in prompt (typically 1.0)"),
    use_blip2: bool = typer.Option(False, help="Use BLIP-2 for dynamic caption generation (better quality, slower)"),
    blip2_model: str = typer.Option("Salesforce/blip2-opt-2.7b", help="BLIP-2 model name (only used if --use-blip2 is set)"),
    combine_with_keywords: bool = typer.Option(True, help="Combine BLIP-2 descriptions with paper keywords (only used if --use-blip2 is set)"),
    use_clip_selection: bool = typer.Option(False, help="Use CLIP to select best template for each image (slower but more accurate, only used if --use-blip2 is False)"),
) -> None:
    """Generate automatic prompts using paper-style keyword format with LoRA weights.

    Paper format: "keyword1, keyword2, ..., defect-specific, loRA:token:weight"
    Example: "grayscale, greyscale, hotrolled steel strip, monochrome, no humans,
             surface defects, texture, rolled-in scale, loRA:neudet1-v1:1"

    With --use-blip2: Uses BLIP-2 to generate dynamic descriptions based on image content,
    optionally combined with paper keywords for better Stable Diffusion compatibility.
    """
    bundle = _ensure_bundle(ctx)
    samples = collect_dataset(dataset_root)
    token_map = {cls: f"<neu_{cls}>" for cls in set(s.cls_name for s in samples)}

    if use_blip2:
        console.print(f"[cyan]Using BLIP-2 model: {blip2_model}[/cyan]")
        console.print("[yellow]BLIP-2 generation is slower but produces more detailed captions[/yellow]")
        captions = generate_captions_with_blip2(
            samples=samples,
            token_map=token_map,
            output_file=output_file,
            use_paper_keywords=use_paper_keywords,
            lora_weight=lora_weight,
            model_name=blip2_model,
            combine_with_keywords=combine_with_keywords,
        )
        console.print(f"Generated {len(captions)} BLIP-2 captions, saved to {output_file}")
        if combine_with_keywords:
            console.print("[cyan]BLIP-2 descriptions combined with paper keywords[/cyan]")
    else:
        generator = CaptionGenerator(model_name=model_name)
        captions = generator.generate_with_token(
            samples,
            token_map,
            output_file=output_file,
            use_paper_keywords=use_paper_keywords,
            lora_weight=lora_weight,
            use_clip_selection=use_clip_selection,
        )
        generator.cleanup()
        console.print(f"Generated {len(captions)} paper-style captions, saved to {output_file}")
        if use_clip_selection:
            console.print("[cyan]Using CLIP to select best template for each image[/cyan]")
        elif use_paper_keywords:
            console.print("[cyan]Using paper-specified keyword format (simple concatenation)[/cyan]")



@app.command("train-drft")
def train_drft_lora(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/drft_lora"), file_okay=False),
    split_manifest: Path = typer.Option(
        _default_split_manifest(),
        "--split-manifest",
        exists=True,
        help="Split manifest; DRFT-LoRA trains only on the train split by default.",
    ),
    train_split: str = typer.Option("train", "--train-split", help="Manifest split used to train DRFT-LoRA."),
    steps: int = typer.Option(100, "--steps", help="DRFT-LoRA training steps."),
    batch_size: int = typer.Option(1, "--batch-size", help="DRFT-LoRA batch size."),
    grad_accum: int = typer.Option(4, "--grad-accum", help="Gradient accumulation steps."),
    learning_rate: float = typer.Option(1e-5, "--learning-rate", help="DRFT-LoRA learning rate."),
    rank: int = typer.Option(4, "--rank", help="LoRA rank for each residual-field expert."),
    alpha: int = typer.Option(4, "--alpha", help="LoRA alpha for each residual-field expert."),
    num_experts: int = typer.Option(3, "--num-experts", help="Number of residual trajectory LoRA experts."),
    max_train_samples: Optional[int] = typer.Option(None, "--max-train-samples", help="Limit samples for smoke training."),
) -> None:
    """Train full DRFT-LoRA on the frozen Stable Diffusion inpainting U-Net."""

    bundle = _ensure_bundle(ctx)
    cfg = replace(bundle.lora)
    cfg.model_id = "runwayml/stable-diffusion-inpainting"
    cfg.steps = max(1, int(steps))
    cfg.batch_size = max(1, int(batch_size))
    cfg.gradient_accumulation_steps = max(1, int(grad_accum))
    cfg.learning_rate = float(learning_rate)
    cfg.rank = max(1, int(rank))
    cfg.alpha = max(1, int(alpha))
    samples = collect_dataset_instances(dataset_root)
    train_stems = _load_split_stems(split_manifest, train_split)
    train_samples = _filter_samples_by_stems(samples, train_stems)
    if not train_samples:
        raise typer.BadParameter(f"No samples matched split '{train_split}' in {split_manifest}")
    console.print(f"[cyan]DRFT-LoRA train split: {train_split} ({len(train_samples)} samples); manifest={split_manifest}[/cyan]")
    token_map = {cls: f"<neu_{cls}>" for cls in sorted({s.cls_name for s in train_samples})}
    hparams = DRFTLoRAHyperParams(rank=cfg.rank, alpha=cfg.alpha, num_experts=max(1, int(num_experts)))
    trainer = DRFTLoRATrainer(cfg, hparams)
    weights = trainer.train(train_samples, token_map, output_dir, max_train_samples=max_train_samples)
    console.print("Saved DRFT-LoRA weights to", weights)


@app.command("train-reference-ldm")
def train_reference_ldm(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/reference_ldm"), file_okay=False),
    split_manifest: Optional[Path] = typer.Option(
        None,
        "--split-manifest",
        help="Optional split_manifest.json. If omitted, split folders under annotations/ are used when present.",
    ),
    train_split: str = typer.Option("train", "--train-split", help="Split used for LDM fine-tuning."),
    base_model: str = typer.Option("runwayml/stable-diffusion-v1-5", "--base-model"),
    per_class: bool = typer.Option(
        True,
        "--per-class/--single-model",
        help="Train one independent LDM per defect class, matching the reference paper.",
    ),
    domain: str = typer.Option("auto", "--domain", help="auto, steel, or fabric."),
    steps: int = typer.Option(1000, "--steps", help="Training steps per model."),
    batch_size: int = typer.Option(1, "--batch-size"),
    grad_accum: int = typer.Option(1, "--grad-accum"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    resolution: int = typer.Option(512, "--resolution"),
    crop_dilation: float = typer.Option(
        1.0,
        "--crop-dilation",
        help="Defect crop dilation used for LDM fine-tuning. The paper uses extracted defect regions.",
    ),
    max_train_samples: Optional[int] = typer.Option(None, "--max-train-samples"),
    mixed_precision: str = typer.Option("no", "--mixed-precision"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Fine-tune the reference-paper LDM on bbox-extracted defect crops."""

    _ = _ensure_bundle(ctx)
    samples = collect_dataset_instances(dataset_root)
    train_samples = _filter_samples_for_split(samples, train_split, split_manifest)
    if not train_samples:
        raise typer.BadParameter(f"No samples matched train split '{train_split}'.")
    effective_domain = infer_surface_domain(dataset_root, domain)
    console.print(
        f"[cyan]Reference LDM training samples: {len(train_samples)}; "
        f"domain={effective_domain}; per_class={per_class}[/cyan]"
    )
    trainer = ReferenceLDMTrainer(
        base_model=base_model,
        resolution=resolution,
        mixed_precision=mixed_precision,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Any] = {
        "method": "reference_ldm_finetune",
        "paper": "Context-aware defect sample generation using conditional diffusion model for surface defect inspection",
        "dataset_root": str(dataset_root.resolve()),
        "train_split": train_split,
        "base_model": base_model,
        "domain": effective_domain,
        "per_class": bool(per_class),
        "models": {},
    }
    if per_class:
        for cls_name in sorted({sample.cls_name for sample in train_samples}):
            class_samples = [sample for sample in train_samples if sample.cls_name == cls_name]
            class_output = output_dir / slugify_class_name(cls_name)
            console.print(f"[cyan]Training class model '{cls_name}' on {len(class_samples)} sample(s).[/cyan]")
            model_path = trainer.train(
                class_samples,
                class_output,
                domain=effective_domain,
                class_name=cls_name,
                steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                grad_accum=grad_accum,
                crop_dilation=crop_dilation,
                max_train_samples=max_train_samples,
                seed=seed,
            )
            index["models"][cls_name] = str(model_path.resolve())
    else:
        model_path = trainer.train(
            train_samples,
            output_dir,
            domain=effective_domain,
            class_name=None,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            grad_accum=grad_accum,
            crop_dilation=crop_dilation,
            max_train_samples=max_train_samples,
            seed=seed,
        )
        index["models"]["all"] = str(model_path.resolve())
    (output_dir / "reference_ldm_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"[green]Reference LDM training artifacts written to {output_dir.resolve()}[/green]")


@app.command("generate-reference-cmdp")
def generate_reference_cmdp(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/reference_cmdp"), file_okay=False),
    split_manifest: Optional[Path] = typer.Option(None, "--split-manifest"),
    generation_split: str = typer.Option("train", "--generation-split"),
    per_class: bool = typer.Option(
        True,
        "--per-class/--single-model",
        help="Use model_dir/<class> when available, matching the paper's per-class LDM setup.",
    ),
    domain: str = typer.Option("auto", "--domain", help="auto, steel, or fabric."),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
    sample_strategy: str = typer.Option("input-order", "--sample-strategy", help="input-order or balanced."),
    dilation_factor: float = typer.Option(1.3, "--dilation-factor", help="Expanded bbox factor gamma used by SCMP."),
    steps: int = typer.Option(50, "--steps", help="Reverse diffusion inference steps."),
    guidance_scale: float = typer.Option(7.0, "--guidance-scale"),
    resolution: int = typer.Option(512, "--resolution"),
    seed: int = typer.Option(42, "--seed"),
    make_mixed_dataset: bool = typer.Option(True, "--make-mixed-dataset/--no-mixed-dataset"),
    orig_images_dir: Optional[Path] = typer.Option(None, "--orig-images-dir", file_okay=False),
    orig_labels_dir: Optional[Path] = typer.Option(None, "--orig-labels-dir", file_okay=False),
    mixed_max_generated: Optional[int] = typer.Option(None, "--mixed-max-generated"),
    mixed_per_source_limit: int = typer.Option(0, "--mixed-per-source-limit"),
    mixed_per_class_limit: int = typer.Option(0, "--mixed-per-class-limit"),
) -> None:
    """Generate samples with the reference paper SCMP + CMDP process."""

    bundle = _ensure_bundle(ctx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_reference-cmdp_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_domain = infer_surface_domain(dataset_root, domain)
    samples = collect_dataset_instances(dataset_root)
    samples = _filter_samples_for_split(samples, generation_split, split_manifest)
    if not samples:
        raise typer.BadParameter(f"No samples matched generation split '{generation_split}'.")
    samples = _limit_samples_by_strategy(samples, max_samples, strategy=sample_strategy)
    console.print(
        f"[cyan]Reference CMDP generation targets: {len(samples)}; "
        f"domain={effective_domain}; per_class={per_class}[/cyan]"
    )

    generated_paths: List[Path] = []
    groups: Dict[str, List[Any]] = {}
    if per_class:
        for sample in samples:
            groups.setdefault(sample.cls_name, []).append(sample)
    else:
        groups["all"] = list(samples)

    for cls_name, cls_samples in groups.items():
        class_model_dir = model_dir / slugify_class_name(cls_name)
        effective_model_dir = class_model_dir if per_class and class_model_dir.exists() else model_dir
        console.print(
            f"[cyan]CMDP group '{cls_name}': {len(cls_samples)} sample(s), model={effective_model_dir}[/cyan]"
        )
        generator = ReferenceCMDPGenerator(effective_model_dir)
        try:
            generated_paths.extend(
                generator.generate(
                    cls_samples,
                    run_dir,
                    domain=effective_domain,
                    dilation_factor=dilation_factor,
                    resolution=resolution,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    max_samples=None,
                )
            )
        finally:
            generator.release()

    final_manifest = {
        "method": "reference_cmdp",
        "paper": "Context-aware defect sample generation using conditional diffusion model for surface defect inspection",
        "dataset_root": str(dataset_root.resolve()),
        "model_dir": str(model_dir.resolve()),
        "generation_split": generation_split,
        "sample_strategy": sample_strategy,
        "domain": effective_domain,
        "generated_images": len(generated_paths),
        "run_dir": str(run_dir.resolve()),
        "image_dir": str((run_dir / "images").resolve()),
        "score_path": str((run_dir / "artifacts" / "reference_cmdp" / "candidate_scores.jsonl").resolve()),
    }
    (run_dir / "reference_cmdp_manifest.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if generated_paths and make_mixed_dataset:
        effective_manifest = split_manifest
        if effective_manifest is None or not Path(effective_manifest).exists():
            effective_manifest = _write_split_manifest_from_voc_layout(dataset_root, run_dir / "split_manifest.json")
        effective_orig_images = orig_images_dir or _default_images_dir_for_dataset(dataset_root)
        effective_orig_labels = orig_labels_dir or _default_labels_dir_for_dataset(dataset_root)
        if not effective_orig_labels.exists():
            console.print(
                f"[yellow]Label directory not found ({effective_orig_labels}); skipping mixed dataset creation.[/yellow]"
            )
        else:
            create_mixed_dataset(
                orig_images_dir=effective_orig_images,
                orig_labels_dir=effective_orig_labels,
                manifest_path=effective_manifest,
                new_images_dir=run_dir / "images",
                run_output_dir=run_dir / "mixed_dataset",
                seed=bundle.dataset.seed,
                score_path=run_dir / "artifacts" / "reference_cmdp" / "candidate_scores.jsonl",
                max_generated=mixed_max_generated,
                per_source_limit=mixed_per_source_limit,
                per_class_limit=mixed_per_class_limit,
                selection_strategy="balanced-quality",
            )
            console.print(f"[green]Mixed dataset written to {(run_dir / 'mixed_dataset').resolve()}[/green]")
    console.print(f"[green]Reference CMDP run written to {run_dir.resolve()}[/green]")


@app.command("generate-dbt-cmdp")
def generate_dbt_cmdp(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/dbt_cmdp"), file_okay=False),
    split_manifest: Optional[Path] = typer.Option(None, "--split-manifest"),
    generation_split: str = typer.Option("train", "--generation-split"),
    per_class: bool = typer.Option(
        True,
        "--per-class/--single-model",
        help="Use model_dir/<class> when available; use --single-model for a base SD snapshot.",
    ),
    domain: str = typer.Option("auto", "--domain", help="auto, steel, or fabric."),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
    sample_strategy: str = typer.Option("input-order", "--sample-strategy", help="input-order or balanced."),
    dilation_factor: float = typer.Option(1.3, "--dilation-factor", help="Expanded bbox factor gamma."),
    steps: int = typer.Option(50, "--steps", help="Reverse diffusion inference steps."),
    guidance_scale: float = typer.Option(7.0, "--guidance-scale"),
    resolution: int = typer.Option(512, "--resolution"),
    seed: int = typer.Option(42, "--seed"),
    eta_min: float = typer.Option(0.30, "--eta-min"),
    eta_max: float = typer.Option(0.85, "--eta-max"),
    dbt_core_radius: int = typer.Option(2, "--dbt-core-radius"),
    dbt_band_radius: int = typer.Option(8, "--dbt-band-radius"),
    dbt_small_core_ratio: float = typer.Option(0.2, "--dbt-small-core-ratio"),
    dbt_out_safety_margin: int = typer.Option(0, "--dbt-out-safety-margin"),
    dbt_residual_blur_sigma: float = typer.Option(2.0, "--dbt-residual-blur-sigma"),
    dbt_gradient_weight: float = typer.Option(0.5, "--dbt-gradient-weight"),
    dbt_tau: Optional[float] = typer.Option(None, "--dbt-tau"),
    dbt_min_area_ratio: float = typer.Option(0.01, "--dbt-min-area-ratio"),
    dbt_max_area_ratio: float = typer.Option(0.25, "--dbt-max-area-ratio"),
    background_propagation_sigma: float = typer.Option(1.0, "--background-propagation-sigma"),
    dbt_mask_geometry: str = typer.Option("pseudo", "--dbt-mask-geometry", help="pseudo or box-adaptive."),
    dbt_box_inner_strength: float = typer.Option(0.65, "--dbt-box-inner-strength"),
    dbt_box_pseudo_strength: float = typer.Option(1.0, "--dbt-box-pseudo-strength"),
    dbt_box_boundary_strength: float = typer.Option(0.50, "--dbt-box-boundary-strength"),
    dbt_stochastic_box: bool = typer.Option(False, "--dbt-stochastic-box/--no-dbt-stochastic-box"),
    dbt_stochastic_profile: str = typer.Option("balanced", "--dbt-stochastic-profile", help="safe, balanced, or open."),
    dbt_shape_preserving: bool = typer.Option(False, "--dbt-shape-preserving/--no-dbt-shape-preserving"),
    dbt_core_generation_strength: float = typer.Option(1.0, "--dbt-core-generation-strength"),
    dbt_core_pixel_strength: float = typer.Option(1.0, "--dbt-core-pixel-strength"),
    dbt_band_pixel_strength: float = typer.Option(0.55, "--dbt-band-pixel-strength"),
    dbt_pixel_residual_clip: Optional[float] = typer.Option(None, "--dbt-pixel-residual-clip"),
    dbt_luminance_only: bool = typer.Option(False, "--dbt-luminance-only/--dbt-rgb-residual"),
    dbt_shape_final_mask: str = typer.Option("defect", "--dbt-shape-final-mask", help="defect or pseudo."),
    dbt_contrast_preservation: float = typer.Option(0.0, "--dbt-contrast-preservation"),
    dbt_contrast_blur_sigma: float = typer.Option(3.0, "--dbt-contrast-blur-sigma"),
    dbt_contrast_min: float = typer.Option(2.5, "--dbt-contrast-min"),
    make_mixed_dataset: bool = typer.Option(True, "--make-mixed-dataset/--no-mixed-dataset"),
    orig_images_dir: Optional[Path] = typer.Option(None, "--orig-images-dir", file_okay=False),
    orig_labels_dir: Optional[Path] = typer.Option(None, "--orig-labels-dir", file_okay=False),
    mixed_max_generated: Optional[int] = typer.Option(None, "--mixed-max-generated"),
    mixed_per_source_limit: int = typer.Option(0, "--mixed-per-source-limit"),
    mixed_per_class_limit: int = typer.Option(0, "--mixed-per-class-limit"),
) -> None:
    """Generate samples with DBT-CMDP tri-domain transition fusion."""

    bundle = _ensure_bundle(ctx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_dbt-cmdp_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_domain = infer_surface_domain(dataset_root, domain)
    samples = collect_dataset_instances(dataset_root)
    samples = _filter_samples_for_split(samples, generation_split, split_manifest)
    if not samples:
        raise typer.BadParameter(f"No samples matched generation split '{generation_split}'.")
    samples = _limit_samples_by_strategy(samples, max_samples, strategy=sample_strategy)
    console.print(
        f"[cyan]DBT-CMDP generation targets: {len(samples)}; "
        f"domain={effective_domain}; per_class={per_class}[/cyan]"
    )

    generated_paths: List[Path] = []
    groups: Dict[str, List[Any]] = {}
    if per_class:
        for sample in samples:
            groups.setdefault(sample.cls_name, []).append(sample)
    else:
        groups["all"] = list(samples)

    for cls_name, cls_samples in groups.items():
        class_model_dir = model_dir / slugify_class_name(cls_name)
        effective_model_dir = class_model_dir if per_class and class_model_dir.exists() else model_dir
        console.print(
            f"[cyan]DBT group '{cls_name}': {len(cls_samples)} sample(s), model={effective_model_dir}[/cyan]"
        )
        generator = DBTCMDPGenerator(effective_model_dir)
        try:
            generated_paths.extend(
                generator.generate(
                    cls_samples,
                    run_dir,
                    domain=effective_domain,
                    dilation_factor=dilation_factor,
                    resolution=resolution,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    max_samples=None,
                    eta_min=eta_min,
                    eta_max=eta_max,
                    dbt_core_radius=dbt_core_radius,
                    dbt_band_radius=dbt_band_radius,
                    dbt_small_core_ratio=dbt_small_core_ratio,
                    dbt_out_safety_margin=dbt_out_safety_margin,
                    dbt_residual_blur_sigma=dbt_residual_blur_sigma,
                    dbt_gradient_weight=dbt_gradient_weight,
                    dbt_tau=dbt_tau,
                    dbt_min_area_ratio=dbt_min_area_ratio,
                    dbt_max_area_ratio=dbt_max_area_ratio,
                    background_propagation_sigma=background_propagation_sigma,
                    dbt_mask_geometry=dbt_mask_geometry,
                    dbt_box_inner_strength=dbt_box_inner_strength,
                    dbt_box_pseudo_strength=dbt_box_pseudo_strength,
                    dbt_box_boundary_strength=dbt_box_boundary_strength,
                    dbt_stochastic_box=dbt_stochastic_box,
                    dbt_stochastic_profile=dbt_stochastic_profile,
                    dbt_shape_preserving=dbt_shape_preserving,
                    dbt_core_generation_strength=dbt_core_generation_strength,
                    dbt_core_pixel_strength=dbt_core_pixel_strength,
                    dbt_band_pixel_strength=dbt_band_pixel_strength,
                    dbt_pixel_residual_clip=dbt_pixel_residual_clip,
                    dbt_luminance_only=dbt_luminance_only,
                    dbt_shape_final_mask=dbt_shape_final_mask,
                    dbt_contrast_preservation=dbt_contrast_preservation,
                    dbt_contrast_blur_sigma=dbt_contrast_blur_sigma,
                    dbt_contrast_min=dbt_contrast_min,
                )
            )
        finally:
            generator.release()

    score_path = run_dir / "artifacts" / "dbt_cmdp" / "candidate_scores.jsonl"
    final_manifest = {
        "method": "dbt_cmdp",
        "base_reference": "Context-aware defect sample generation using conditional diffusion model for surface defect inspection",
        "dataset_root": str(dataset_root.resolve()),
        "model_dir": str(model_dir.resolve()),
        "generation_split": generation_split,
        "sample_strategy": sample_strategy,
        "domain": effective_domain,
        "generated_images": len(generated_paths),
        "run_dir": str(run_dir.resolve()),
        "image_dir": str((run_dir / "images").resolve()),
        "score_path": str(score_path.resolve()),
        "dbt": {
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
            "mask_geometry": str(dbt_mask_geometry),
            "box_inner_strength": float(dbt_box_inner_strength),
            "box_pseudo_strength": float(dbt_box_pseudo_strength),
            "box_boundary_strength": float(dbt_box_boundary_strength),
            "stochastic_box": bool(dbt_stochastic_box),
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
        },
    }
    (run_dir / "dbt_cmdp_manifest.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if generated_paths and make_mixed_dataset:
        effective_manifest = split_manifest
        if effective_manifest is None or not Path(effective_manifest).exists():
            effective_manifest = _write_split_manifest_from_voc_layout(dataset_root, run_dir / "split_manifest.json")
        effective_orig_images = orig_images_dir or _default_images_dir_for_dataset(dataset_root)
        effective_orig_labels = orig_labels_dir or _default_labels_dir_for_dataset(dataset_root)
        if not effective_orig_labels.exists():
            console.print(
                f"[yellow]Label directory not found ({effective_orig_labels}); skipping mixed dataset creation.[/yellow]"
            )
        else:
            create_mixed_dataset(
                orig_images_dir=effective_orig_images,
                orig_labels_dir=effective_orig_labels,
                manifest_path=effective_manifest,
                new_images_dir=run_dir / "images",
                run_output_dir=run_dir / "mixed_dataset",
                seed=bundle.dataset.seed,
                score_path=score_path,
                max_generated=mixed_max_generated,
                per_source_limit=mixed_per_source_limit,
                per_class_limit=mixed_per_class_limit,
                selection_strategy="balanced-quality",
            )
            console.print(f"[green]Mixed dataset written to {(run_dir / 'mixed_dataset').resolve()}[/green]")
    console.print(f"[green]DBT-CMDP run written to {run_dir.resolve()}[/green]")


@app.command("generate-context-sgda")
def generate_context_sgda(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/context_sgda"), file_okay=False),
    split_manifest: Optional[Path] = typer.Option(None, "--split-manifest"),
    generation_split: str = typer.Option("train", "--generation-split"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
    sample_strategy: str = typer.Option("balanced", "--sample-strategy", help="input-order or balanced."),
    target_size: int = typer.Option(512, "--target-size", help="Square working resolution for context-coordinate residual fields."),
    steps: int = typer.Option(18, "--steps", help="Residual-flow integration steps."),
    candidates: int = typer.Option(2, "--candidates", help="Structured residual candidates sampled per target."),
    seed: int = typer.Option(42, "--seed"),
    context_dilation: float = typer.Option(1.65, "--context-dilation"),
    context_shell_weight: float = typer.Option(0.55, "--context-shell-weight"),
    context_core_weight: float = typer.Option(1.0, "--context-core-weight"),
    context_core_threshold: float = typer.Option(0.10, "--context-core-threshold"),
    feather_radius: float = typer.Option(1.4, "--feather-radius"),
    make_mixed_dataset: bool = typer.Option(True, "--make-mixed-dataset/--no-mixed-dataset"),
    orig_images_dir: Optional[Path] = typer.Option(None, "--orig-images-dir", file_okay=False),
    orig_labels_dir: Optional[Path] = typer.Option(None, "--orig-labels-dir", file_okay=False),
    mixed_max_generated: Optional[int] = typer.Option(None, "--mixed-max-generated"),
    mixed_per_source_limit: int = typer.Option(0, "--mixed-per-source-limit"),
    mixed_per_class_limit: int = typer.Option(0, "--mixed-per-class-limit"),
    mixed_quality_threshold: Optional[float] = typer.Option(None, "--mixed-quality-threshold"),
    mixed_max_outside_delta: Optional[float] = typer.Option(None, "--mixed-max-outside-delta"),
) -> None:
    """Generate Context-SGDA samples with structured residual fields and context contracts."""

    bundle = _ensure_bundle(ctx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_context-sgda_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_dataset_instances(dataset_root)
    samples = _filter_samples_for_split(samples, generation_split, split_manifest)
    if not samples:
        raise typer.BadParameter(f"No samples matched generation split '{generation_split}'.")
    reference_samples = list(samples)
    samples = _limit_samples_by_strategy(samples, max_samples, strategy=sample_strategy)
    console.print(
        f"[cyan]Context-SGDA targets: {len(samples)}; split={generation_split}; "
        f"strategy={sample_strategy}; candidates={max(1, int(candidates))}[/cyan]"
    )

    cfg = ContextSGDAConfig(
        target_size=(int(target_size), int(target_size)),
        steps=max(3, int(steps)),
        candidates_per_sample=max(1, int(candidates)),
        seed=int(seed),
        context_dilation=float(context_dilation),
        context_shell_weight=float(context_shell_weight),
        context_core_weight=float(context_core_weight),
        context_core_threshold=float(context_core_threshold),
        feather_radius=float(feather_radius),
    )
    generator = ContextSGDAGenerator(cfg)
    generated_paths = generator.generate(samples, run_dir, reference_samples=reference_samples)
    score_path = run_dir / "artifacts" / "context_sgda" / "candidate_scores.jsonl"

    final_manifest = {
        "method": "context_sgda_residual",
        "dataset_root": str(dataset_root.resolve()),
        "generation_split": generation_split,
        "sample_strategy": sample_strategy,
        "generated_images": len(generated_paths),
        "run_dir": str(run_dir.resolve()),
        "image_dir": str((run_dir / "images").resolve()),
        "score_path": str(score_path.resolve()),
        "config": {
            "target_size": int(target_size),
            "steps": int(steps),
            "candidates": int(candidates),
            "seed": int(seed),
            "context_dilation": float(context_dilation),
            "context_shell_weight": float(context_shell_weight),
            "context_core_weight": float(context_core_weight),
            "context_core_threshold": float(context_core_threshold),
            "feather_radius": float(feather_radius),
        },
        "innovation": [
            "context-coordinate generation through spatial, morphology and texture codes",
            "context-preserved residual generation using I_syn = I_real + A(context, box) * delta_defect",
        ],
    }
    (run_dir / "context_sgda_run_manifest.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if generated_paths and make_mixed_dataset:
        effective_manifest = split_manifest
        if effective_manifest is None or not Path(effective_manifest).exists():
            effective_manifest = _write_split_manifest_from_voc_layout(dataset_root, run_dir / "split_manifest.json")
        effective_orig_images = orig_images_dir or _default_images_dir_for_dataset(dataset_root)
        effective_orig_labels = orig_labels_dir or _default_labels_dir_for_dataset(dataset_root)
        if not effective_orig_labels.exists():
            console.print(
                f"[yellow]Label directory not found ({effective_orig_labels}); skipping mixed dataset creation.[/yellow]"
            )
        else:
            create_mixed_dataset(
                orig_images_dir=effective_orig_images,
                orig_labels_dir=effective_orig_labels,
                manifest_path=effective_manifest,
                new_images_dir=run_dir / "images",
                run_output_dir=run_dir / "mixed_dataset",
                seed=bundle.dataset.seed,
                score_path=score_path,
                max_generated=mixed_max_generated,
                per_source_limit=mixed_per_source_limit,
                per_class_limit=mixed_per_class_limit,
                quality_threshold=mixed_quality_threshold,
                max_outside_delta=mixed_max_outside_delta,
                selection_strategy="balanced-quality",
            )
            console.print(f"[green]Mixed dataset written to {(run_dir / 'mixed_dataset').resolve()}[/green]")
    console.print(f"[green]Context-SGDA run written to {run_dir.resolve()}[/green]")


@app.command("generate-context-lgi")
def generate_context_lgi(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/context_lgi"), file_okay=False),
    split_manifest: Optional[Path] = typer.Option(None, "--split-manifest"),
    generation_split: str = typer.Option("train", "--generation-split"),
    per_class: bool = typer.Option(
        True,
        "--per-class/--single-model",
        help="Use model_dir/<class> when available; use --single-model for one LDM.",
    ),
    domain: str = typer.Option("auto", "--domain", help="auto, steel, or fabric."),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
    sample_strategy: str = typer.Option("balanced", "--sample-strategy", help="input-order or balanced."),
    resolution: int = typer.Option(512, "--resolution"),
    steps: int = typer.Option(30, "--steps", help="Reverse diffusion inference steps."),
    guidance_scale: float = typer.Option(7.0, "--guidance-scale"),
    seed: int = typer.Option(42, "--seed"),
    candidates: int = typer.Option(1, "--candidates", help="Structured latent candidates per target."),
    dilation_factor: float = typer.Option(1.35, "--dilation-factor"),
    eta_min: float = typer.Option(0.20, "--eta-min"),
    eta_max: float = typer.Option(0.78, "--eta-max"),
    shell_strength: float = typer.Option(0.55, "--shell-strength"),
    core_strength: float = typer.Option(1.0, "--core-strength"),
    background_propagation_sigma: float = typer.Option(1.0, "--background-propagation-sigma"),
    feather_radius: float = typer.Option(1.2, "--feather-radius"),
    make_mixed_dataset: bool = typer.Option(True, "--make-mixed-dataset/--no-mixed-dataset"),
    orig_images_dir: Optional[Path] = typer.Option(None, "--orig-images-dir", file_okay=False),
    orig_labels_dir: Optional[Path] = typer.Option(None, "--orig-labels-dir", file_okay=False),
    mixed_max_generated: Optional[int] = typer.Option(None, "--mixed-max-generated"),
    mixed_per_source_limit: int = typer.Option(0, "--mixed-per-source-limit"),
    mixed_per_class_limit: int = typer.Option(0, "--mixed-per-class-limit"),
    mixed_quality_threshold: Optional[float] = typer.Option(None, "--mixed-quality-threshold"),
    mixed_max_outside_delta: Optional[float] = typer.Option(None, "--mixed-max-outside-delta"),
) -> None:
    """Generate non-residual Context-LGI samples with context-preserved latent inpainting."""

    bundle = _ensure_bundle(ctx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_context-lgi_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_domain = infer_surface_domain(dataset_root, domain)

    all_samples = collect_dataset_instances(dataset_root)
    split_samples = _filter_samples_for_split(all_samples, generation_split, split_manifest)
    if not split_samples:
        raise typer.BadParameter(f"No samples matched generation split '{generation_split}'.")
    samples = _limit_samples_by_strategy(split_samples, max_samples, strategy=sample_strategy)
    console.print(
        f"[cyan]Context-LGI targets: {len(samples)}; split={generation_split}; "
        f"domain={effective_domain}; per_class={per_class}; candidates={max(1, int(candidates))}[/cyan]"
    )

    cfg = ContextLGIConfig(
        resolution=int(resolution),
        dilation_factor=float(dilation_factor),
        num_inference_steps=max(4, int(steps)),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        candidates_per_sample=max(1, int(candidates)),
        eta_min=float(eta_min),
        eta_max=float(eta_max),
        shell_strength=float(shell_strength),
        core_strength=float(core_strength),
        background_propagation_sigma=float(background_propagation_sigma),
        feather_radius=float(feather_radius),
    )

    sample_groups: Dict[str, List[Any]] = {}
    if per_class:
        for sample in samples:
            sample_groups.setdefault(sample.cls_name, []).append(sample)
    else:
        sample_groups["all"] = list(samples)

    reference_groups: Dict[str, List[Any]] = {}
    if per_class:
        for sample in split_samples:
            reference_groups.setdefault(sample.cls_name, []).append(sample)
    else:
        reference_groups["all"] = list(split_samples)

    generated_paths: List[Path] = []
    for cls_name, cls_samples in sample_groups.items():
        class_model_dir = model_dir / slugify_class_name(cls_name)
        effective_model_dir = class_model_dir if per_class and class_model_dir.exists() else model_dir
        refs = reference_groups.get(cls_name, cls_samples) if per_class else reference_groups["all"]
        console.print(
            f"[cyan]Context-LGI group '{cls_name}': {len(cls_samples)} sample(s), model={effective_model_dir}[/cyan]"
        )
        generator = ContextLGIGenerator(effective_model_dir)
        try:
            generated_paths.extend(
                generator.generate_lgi(
                    cls_samples,
                    run_dir,
                    domain=effective_domain,
                    reference_samples=refs,
                    config=cfg,
                )
            )
        finally:
            generator.release()

    score_path = run_dir / "artifacts" / "context_lgi" / "candidate_scores.jsonl"
    final_manifest = {
        "method": "context_lgi",
        "dataset_root": str(dataset_root.resolve()),
        "model_dir": str(model_dir.resolve()),
        "generation_split": generation_split,
        "sample_strategy": sample_strategy,
        "domain": effective_domain,
        "per_class": bool(per_class),
        "generated_images": len(generated_paths),
        "run_dir": str(run_dir.resolve()),
        "image_dir": str((run_dir / "images").resolve()),
        "score_path": str(score_path.resolve()),
        "config": asdict(cfg),
        "innovation": [
            "context-coordinate generation through structured space, shape and texture latent conditions",
            "context-preserved latent inpainting with step-wise M_out background injection and M_shell transition control",
        ],
    }
    (run_dir / "context_lgi_run_manifest.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if generated_paths and make_mixed_dataset:
        effective_manifest = split_manifest
        if effective_manifest is None or not Path(effective_manifest).exists():
            effective_manifest = _write_split_manifest_from_voc_layout(dataset_root, run_dir / "split_manifest.json")
        effective_orig_images = orig_images_dir or _default_images_dir_for_dataset(dataset_root)
        effective_orig_labels = orig_labels_dir or _default_labels_dir_for_dataset(dataset_root)
        if not effective_orig_labels.exists():
            console.print(
                f"[yellow]Label directory not found ({effective_orig_labels}); skipping mixed dataset creation.[/yellow]"
            )
        else:
            create_mixed_dataset(
                orig_images_dir=effective_orig_images,
                orig_labels_dir=effective_orig_labels,
                manifest_path=effective_manifest,
                new_images_dir=run_dir / "images",
                run_output_dir=run_dir / "mixed_dataset",
                seed=bundle.dataset.seed,
                score_path=score_path,
                max_generated=mixed_max_generated,
                per_source_limit=mixed_per_source_limit,
                per_class_limit=mixed_per_class_limit,
                quality_threshold=mixed_quality_threshold,
                max_outside_delta=mixed_max_outside_delta,
                selection_strategy="balanced-quality",
            )
            console.print(f"[green]Mixed dataset written to {(run_dir / 'mixed_dataset').resolve()}[/green]")
    console.print(f"[green]Context-LGI run written to {run_dir.resolve()}[/green]")


@app.command("generate-de-lgi")
def generate_de_lgi(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/de_lgi"), file_okay=False),
    split_manifest: Optional[Path] = typer.Option(None, "--split-manifest"),
    generation_split: str = typer.Option("train", "--generation-split"),
    per_class: bool = typer.Option(
        True,
        "--per-class/--single-model",
        help="Use model_dir/<class> when available; use --single-model for one LDM.",
    ),
    domain: str = typer.Option("auto", "--domain", help="auto, steel, or fabric."),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
    target_generated_images: Optional[int] = typer.Option(
        None,
        "--target-generated-images",
        help="Requested generated-image count; Spec-SRT expands the target pool and writes a curated balanced subset when possible.",
    ),
    sample_strategy: str = typer.Option("balanced", "--sample-strategy", help="input-order or balanced."),
    preset: str = typer.Option("custom", "--preset", help="Generation preset: custom, spec-srt, hdsi-s2c, or e-srt-hdsi-pd."),
    resolution: int = typer.Option(512, "--resolution"),
    steps: int = typer.Option(30, "--steps", help="Reverse diffusion inference steps."),
    guidance_scale: float = typer.Option(7.0, "--guidance-scale"),
    seed: int = typer.Option(42, "--seed"),
    candidates: int = typer.Option(1, "--candidates", help="Energy-projected latent candidates per target."),
    dilation_factor: float = typer.Option(1.35, "--dilation-factor"),
    eta_min: float = typer.Option(0.20, "--eta-min"),
    eta_max: float = typer.Option(0.78, "--eta-max"),
    shell_strength: float = typer.Option(0.55, "--shell-strength"),
    background_leakage: float = typer.Option(0.03, "--background-leakage"),
    background_propagation_sigma: float = typer.Option(1.0, "--background-propagation-sigma"),
    boundary_smoothing_sigma: float = typer.Option(1.15, "--boundary-smoothing-sigma"),
    feather_radius: float = typer.Option(1.2, "--feather-radius"),
    latent_energy_scale: float = typer.Option(8.0, "--latent-energy-scale"),
    latent_energy_floor: float = typer.Option(0.12, "--latent-energy-floor"),
    latent_energy_ceiling: float = typer.Option(0.50, "--latent-energy-ceiling"),
    energy_jitter: float = typer.Option(0.16, "--energy-jitter"),
    min_projection_scale: float = typer.Option(0.12, "--min-projection-scale"),
    max_projection_scale: float = typer.Option(3.40, "--max-projection-scale"),
    use_ucs: bool = typer.Option(True, "--ucs/--classic-de-lgi", help="Use utility-guided causal spectrum projection."),
    spectrum_jitter: float = typer.Option(0.22, "--spectrum-jitter"),
    utility_guidance: float = typer.Option(0.45, "--utility-guidance"),
    spectrum_projection_strength: float = typer.Option(0.82, "--spectrum-projection-strength"),
    spectrum_orientation_strength: float = typer.Option(0.28, "--spectrum-orientation-strength"),
    core_renoise_strength: float = typer.Option(0.18, "--core-renoise-strength"),
    diversity_weight: float = typer.Option(0.12, "--diversity-weight"),
    use_drr: bool = typer.Option(False, "--drr/--no-drr", help="Use diversity-aware cross-instance residual recomposition."),
    residual_bank_mix: float = typer.Option(0.48, "--residual-bank-mix"),
    pseudo_suppression_strength: float = typer.Option(0.62, "--pseudo-suppression-strength"),
    structure_jitter: float = typer.Option(0.32, "--structure-jitter"),
    diversity_delta_target: float = typer.Option(0.070, "--diversity-delta-target"),
    outside_delta_budget: float = typer.Option(0.014, "--outside-delta-budget"),
    use_distant_spectrum: bool = typer.Option(False, "--distant-spectrum/--no-distant-spectrum", help="Sample class-valid target spectra far from the source sample spectrum."),
    distant_spectrum_candidates: int = typer.Option(8, "--distant-spectrum-candidates"),
    distant_spectrum_weight: float = typer.Option(0.70, "--distant-spectrum-weight"),
    use_hdsi: bool = typer.Option(False, "--hdsi/--no-hdsi", help="Use hardness-aware defect spectrum intervention for hard-but-valid target spectra."),
    hdsi_candidates: int = typer.Option(12, "--hdsi-candidates", help="Candidate target spectra sampled for HDSI selection."),
    hdsi_hardness_weight: float = typer.Option(0.58, "--hdsi-hardness-weight", help="HDSI weight for hard defect-spectrum traits."),
    hdsi_validity_weight: float = typer.Option(0.30, "--hdsi-validity-weight", help="HDSI gate weight for class-prior validity."),
    hdsi_diversity_weight: float = typer.Option(0.42, "--hdsi-diversity-weight", help="HDSI weight for source/history spectral diversity."),
    hdsi_projection_boost: float = typer.Option(0.32, "--hdsi-projection-boost", help="Projection boost from the HDSI intervention score."),
    hdsi_tail_strength: float = typer.Option(0.55, "--hdsi-tail-strength", help="Strength of HDSI high-frequency/roughness tail intervention."),
    hdsi_min_validity_score: float = typer.Option(0.38, "--hdsi-min-validity-score", help="Minimum class-prior validity score for HDSI target spectra."),
    hdsi_min_source_distance: float = typer.Option(0.12, "--hdsi-min-source-distance", help="Minimum source-spectrum distance for HDSI target spectra when source spectrum is available."),
    use_hdsi_s2c: bool = typer.Option(False, "--hdsi-s2c/--no-hdsi-s2c", help="Use HDSI spectrum-to-structure/image consistency scoring and gating."),
    hdsi_s2c_structure_weight: float = typer.Option(0.34, "--hdsi-s2c-structure-weight", help="Weight for target-spectrum realization in phi'."),
    hdsi_s2c_image_weight: float = typer.Option(0.46, "--hdsi-s2c-image-weight", help="Weight for target-spectrum realization in generated pixels."),
    hdsi_s2c_min_score: float = typer.Option(0.42, "--hdsi-s2c-min-score", help="Minimum image-side HDSI-S2C score before strict rejection."),
    hdsi_s2c_strict_gate: bool = typer.Option(True, "--hdsi-s2c-strict-gate/--hdsi-s2c-soft-gate", help="Reject candidates failing HDSI-S2C consistency."),
    hdsi_s2c_spectrum_tolerance: float = typer.Option(0.34, "--hdsi-s2c-spectrum-tolerance", help="Tolerance for generated-vs-target defect spectrum matching."),
    use_hdsi_pd: bool = typer.Option(False, "--hdsi-pd/--no-hdsi-pd", help="Use HDSI-PD spectrum-phase prototype detail intervention on top of SRT+HDSI."),
    hdsi_pd_prototype_count: int = typer.Option(2, "--hdsi-pd-prototypes", help="Same-class real defect prototypes used for HDSI-PD phase/detail injection."),
    hdsi_pd_strength: float = typer.Option(0.44, "--hdsi-pd-strength", help="Strength for mixing HDSI-PD prototype residual into SRT residual transport."),
    hdsi_pd_phase_strength: float = typer.Option(0.42, "--hdsi-pd-phase-strength", help="Orientation/phase shaping strength for HDSI-PD prototype detail."),
    hdsi_pd_late_detail_strength: float = typer.Option(0.34, "--hdsi-pd-late-detail-strength", help="Late denoising high-frequency detail release inside the defect core."),
    hdsi_pd_detail_start: float = typer.Option(0.58, "--hdsi-pd-detail-start", help="Fraction of denoising steps after which HDSI-PD detail is released."),
    hdsi_pd_late_renoise_strength: float = typer.Option(0.10, "--hdsi-pd-late-renoise-strength", help="Late high-frequency noise used to avoid prototype over-copying."),
    use_band_recomposition: bool = typer.Option(False, "--band-recomposition/--no-band-recomposition", help="Compose low/mid/high latent residual bands from multiple donors."),
    band_donor_count: int = typer.Option(3, "--band-donor-count"),
    band_recomposition_strength: float = typer.Option(0.70, "--band-recomposition-strength"),
    use_srt: bool = typer.Option(False, "--srt/--no-srt", help="Use structure-residual transport with source bbox labels by default."),
    srt_transport_strength: float = typer.Option(0.72, "--srt-transport-strength"),
    srt_bbox_jitter: float = typer.Option(0.30, "--srt-bbox-jitter"),
    srt_scale_jitter: float = typer.Option(0.34, "--srt-scale-jitter"),
    srt_boundary_roughness: float = typer.Option(0.58, "--srt-boundary-roughness"),
    srt_component_jitter: float = typer.Option(0.55, "--srt-component-jitter"),
    srt_source_preservation: float = typer.Option(0.28, "--srt-source-preservation"),
    srt_bbox_update: bool = typer.Option(False, "--srt-bbox-update/--srt-inherit-bbox", help="Derive SRT labels from generated structure; default inherits source bbox."),
    srt_regeneration_strength: float = typer.Option(0.42, "--srt-regeneration-strength", help="Early diffusion strength for phi'-guided local regeneration."),
    srt_regeneration_until: float = typer.Option(0.45, "--srt-regeneration-until", help="Fraction of diffusion steps that use strong SRT structure anchoring."),
    srt_late_texture_mix: float = typer.Option(0.34, "--srt-late-texture-mix", help="Late-step transported residual mix kept for texture repair."),
    srt_s2i_visible_delta_target: float = typer.Option(0.018, "--srt-s2i-visible-delta-target", help="Target visible image delta on phi' for SRT candidate selection."),
    srt_s2i_min_ratio: float = typer.Option(1.35, "--srt-s2i-min-ratio", help="Minimum on-phi/off-phi image-change ratio for SRT candidate selection."),
    use_e_srt: bool = typer.Option(False, "--e-srt/--no-e-srt", help="Use evidence-guided SRT feedback during denoising."),
    e_srt_evidence_strength: float = typer.Option(0.34, "--e-srt-evidence-strength", help="Strength for boosting weak structure-region evidence."),
    e_srt_background_strength: float = typer.Option(0.42, "--e-srt-background-strength", help="Strength for suppressing background evidence leakage."),
    e_srt_min_core_energy_ratio: float = typer.Option(0.72, "--e-srt-min-core-energy-ratio", help="Minimum target-energy ratio expected inside phi'."),
    use_seca: bool = typer.Option(False, "--seca/--no-seca", help="Derive labels from structure-evidence agreement after generation."),
    seca_min_coverage: float = typer.Option(0.18, "--seca-min-coverage", help="Minimum structure coverage required for SECA pass."),
    seca_max_leakage: float = typer.Option(0.34, "--seca-max-leakage", help="Maximum off-structure evidence leakage allowed for SECA pass."),
    seca_min_focus: float = typer.Option(1.18, "--seca-min-focus", help="Minimum on-structure/off-structure evidence ratio for SECA pass."),
    seca_strict: bool = typer.Option(True, "--seca-strict/--seca-soft", help="Reject candidates failing SECA; --seca-soft keeps soft down-ranking."),
    srt_strict_s2i_gate: bool = typer.Option(True, "--srt-strict-s2i-gate/--srt-soft-s2i-gate", help="Reject SRT candidates failing structure-to-image consistency; --srt-soft-s2i-gate keeps soft down-ranking."),
    make_mixed_dataset: bool = typer.Option(True, "--make-mixed-dataset/--no-mixed-dataset"),
    orig_images_dir: Optional[Path] = typer.Option(None, "--orig-images-dir", file_okay=False),
    orig_labels_dir: Optional[Path] = typer.Option(None, "--orig-labels-dir", file_okay=False),
    mixed_max_generated: Optional[int] = typer.Option(None, "--mixed-max-generated"),
    mixed_per_source_limit: int = typer.Option(0, "--mixed-per-source-limit"),
    mixed_per_class_limit: int = typer.Option(0, "--mixed-per-class-limit"),
    mixed_quality_threshold: Optional[float] = typer.Option(None, "--mixed-quality-threshold"),
    mixed_max_outside_delta: Optional[float] = typer.Option(None, "--mixed-max-outside-delta"),
) -> None:
    """Generate DE-LGI samples with class-calibrated latent residual energy projection."""

    bundle = _ensure_bundle(ctx)
    effective_domain = infer_surface_domain(dataset_root, domain)
    preset_key = (preset or "custom").strip().lower().replace("_", "-").replace("+", "-")
    if preset_key == "esrt-hdsi-pd":
        preset_key = "e-srt-hdsi-pd"
    if preset_key in {"hdsis2c", "hdsi-s2c-de-lgi"}:
        preset_key = "hdsi-s2c"
    if preset_key not in {"custom", "spec-srt", "hdsi-s2c", "e-srt-hdsi-pd"}:
        raise typer.BadParameter("--preset must be 'custom', 'spec-srt', 'hdsi-s2c', or 'e-srt-hdsi-pd'.")
    if preset_key == "spec-srt":
        use_ucs = True
        use_srt = True
        use_e_srt = True
        use_hdsi = True
        use_distant_spectrum = True
        use_drr = False
        use_band_recomposition = False
        use_seca = False
        use_hdsi_pd = False
        srt_bbox_update = False
        band_donor_count = 1
        srt_strict_s2i_gate = True
        candidates = max(int(candidates), 8)
        diversity_weight = max(float(diversity_weight), 0.40)
        core_renoise_strength = max(float(core_renoise_strength), 0.40)
        pseudo_suppression_strength = max(float(pseudo_suppression_strength), 0.58)
        boundary_smoothing_sigma = min(float(boundary_smoothing_sigma), 0.45)
        feather_radius = min(float(feather_radius), 0.50)
        background_propagation_sigma = min(float(background_propagation_sigma), 0.58)
        e_srt_background_strength = min(float(e_srt_background_strength), 0.24)
        srt_s2i_min_ratio = float(srt_s2i_min_ratio)
        if srt_s2i_min_ratio <= 1.35:
            srt_s2i_min_ratio = 0.95
        outside_delta_budget = max(float(outside_delta_budget), 0.016)
        distant_spectrum_candidates = max(int(distant_spectrum_candidates), 32)
        hdsi_candidates = max(int(hdsi_candidates), 32)
        distant_spectrum_weight = max(float(distant_spectrum_weight), 0.82)
        spectrum_projection_strength = max(float(spectrum_projection_strength), 0.90)
        hdsi_tail_strength = max(float(hdsi_tail_strength), 0.72)
        structure_jitter = max(float(structure_jitter), 0.45)
        srt_transport_strength = max(float(srt_transport_strength), 0.82)
        srt_scale_jitter = max(float(srt_scale_jitter), 0.42)
        srt_boundary_roughness = max(float(srt_boundary_roughness), 0.68)
        srt_component_jitter = max(float(srt_component_jitter), 0.70)
        srt_source_preservation = min(float(srt_source_preservation), 0.14)
        srt_late_texture_mix = max(float(srt_late_texture_mix), 0.60)
    if preset_key == "hdsi-s2c":
        use_ucs = True
        use_srt = True
        use_e_srt = True
        use_hdsi = True
        use_hdsi_s2c = True
        use_hdsi_pd = False
        use_drr = False
        use_band_recomposition = False
        use_seca = False
        srt_bbox_update = False
        band_donor_count = 1
        candidates = max(int(candidates), 4)
        guidance_scale = min(float(guidance_scale), 6.2)
        background_leakage = min(float(background_leakage), 0.024)
        boundary_smoothing_sigma = min(float(boundary_smoothing_sigma), 0.62)
        feather_radius = min(float(feather_radius), 0.68)
        core_renoise_strength = max(float(core_renoise_strength), 0.42)
        diversity_weight = max(float(diversity_weight), 0.40)
        pseudo_suppression_strength = min(float(pseudo_suppression_strength), 0.54)
        outside_delta_budget = max(float(outside_delta_budget), 0.016)
        use_distant_spectrum = True
        distant_spectrum_candidates = max(int(distant_spectrum_candidates), 24)
        hdsi_candidates = max(int(hdsi_candidates), 24)
        distant_spectrum_weight = max(float(distant_spectrum_weight), 0.84)
        spectrum_projection_strength = max(float(spectrum_projection_strength), 0.92)
        spectrum_orientation_strength = max(float(spectrum_orientation_strength), 0.34)
        hdsi_tail_strength = max(float(hdsi_tail_strength), 0.74)
        hdsi_min_validity_score = max(float(hdsi_min_validity_score), 0.50)
        hdsi_min_source_distance = max(float(hdsi_min_source_distance), 0.20)
        srt_transport_strength = max(float(srt_transport_strength), 0.84)
        srt_bbox_jitter = max(float(srt_bbox_jitter), 0.46)
        srt_scale_jitter = max(float(srt_scale_jitter), 0.50)
        srt_boundary_roughness = max(float(srt_boundary_roughness), 0.74)
        srt_component_jitter = max(float(srt_component_jitter), 0.76)
        srt_source_preservation = min(float(srt_source_preservation), 0.10)
        srt_late_texture_mix = max(float(srt_late_texture_mix), 0.64)
        srt_strict_s2i_gate = False
        hdsi_s2c_structure_weight = max(float(hdsi_s2c_structure_weight), 0.34)
        hdsi_s2c_image_weight = max(float(hdsi_s2c_image_weight), 0.46)
        hdsi_s2c_min_score = max(float(hdsi_s2c_min_score), 0.40)
        hdsi_s2c_spectrum_tolerance = min(float(hdsi_s2c_spectrum_tolerance), 0.36)
    if preset_key == "e-srt-hdsi-pd":
        use_ucs = True
        use_srt = True
        use_e_srt = True
        use_hdsi = True
        use_hdsi_pd = True
        use_drr = False
        use_band_recomposition = False
        use_seca = False
        srt_bbox_update = False
        band_donor_count = 1
        candidates = max(int(candidates), 5)
        guidance_scale = min(float(guidance_scale), 6.0)
        background_leakage = min(float(background_leakage), 0.025)
        boundary_smoothing_sigma = min(float(boundary_smoothing_sigma), 0.75)
        feather_radius = min(float(feather_radius), 0.80)
        core_renoise_strength = max(float(core_renoise_strength), 0.42)
        diversity_weight = max(float(diversity_weight), 0.42)
        pseudo_suppression_strength = min(float(pseudo_suppression_strength), 0.50)
        outside_delta_budget = max(float(outside_delta_budget), 0.016)
        use_distant_spectrum = True
        distant_spectrum_candidates = max(int(distant_spectrum_candidates), 24)
        hdsi_candidates = max(int(hdsi_candidates), 24)
        distant_spectrum_weight = max(float(distant_spectrum_weight), 0.82)
        hdsi_tail_strength = max(float(hdsi_tail_strength), 0.72)
        srt_transport_strength = max(float(srt_transport_strength), 0.84)
        srt_bbox_jitter = max(float(srt_bbox_jitter), 0.58)
        srt_scale_jitter = max(float(srt_scale_jitter), 0.62)
        srt_boundary_roughness = max(float(srt_boundary_roughness), 0.76)
        srt_component_jitter = max(float(srt_component_jitter), 0.78)
        srt_source_preservation = min(float(srt_source_preservation), 0.08)
        srt_late_texture_mix = max(float(srt_late_texture_mix), 0.66)
        hdsi_pd_prototype_count = max(int(hdsi_pd_prototype_count), 3)
        hdsi_pd_strength = max(float(hdsi_pd_strength), 0.62)
        hdsi_pd_phase_strength = max(float(hdsi_pd_phase_strength), 0.55)
        hdsi_pd_late_detail_strength = max(float(hdsi_pd_late_detail_strength), 0.46)
        hdsi_pd_late_renoise_strength = max(float(hdsi_pd_late_renoise_strength), 0.16)
        # HDSI-PD deliberately releases late high-frequency detail; keep S2I
        # as a ranking signal instead of a hard annotation veto for this preset.
        srt_strict_s2i_gate = False
    if (use_e_srt or use_seca) and not (use_srt and use_ucs):
        raise typer.BadParameter("--e-srt and --seca require --ucs --srt because they extend SRT-DE-LGI.")
    if use_hdsi and not use_ucs:
        raise typer.BadParameter("--hdsi requires --ucs because it extends utility-guided causal spectrum projection.")
    if use_hdsi_s2c and not (use_ucs and use_srt and use_hdsi):
        raise typer.BadParameter("--hdsi-s2c requires --ucs --srt --hdsi.")
    if use_hdsi_pd and not (use_ucs and use_srt and use_hdsi):
        raise typer.BadParameter("--hdsi-pd requires --ucs --srt --hdsi because it injects prototype details into SRT+HDSI.")
    is_hdsi_s2c_run = bool(use_ucs and use_srt and use_hdsi and use_hdsi_s2c)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_de-lgi_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_samples = collect_dataset_instances(dataset_root)
    split_samples = _filter_samples_for_split(all_samples, generation_split, split_manifest)
    if not split_samples:
        raise typer.BadParameter(f"No samples matched generation split '{generation_split}'.")
    requested_generated = int(target_generated_images) if target_generated_images is not None else None
    if requested_generated is not None and requested_generated <= 0:
        raise typer.BadParameter("--target-generated-images must be positive when provided.")
    effective_max_samples = max_samples
    if preset_key == "spec-srt" and requested_generated is not None:
        expanded_pool = min(len(split_samples), max(requested_generated * 3, requested_generated))
        effective_max_samples = max(int(max_samples or 0), expanded_pool) if max_samples is not None else expanded_pool
    elif (is_hdsi_s2c_run or preset_key == "e-srt-hdsi-pd") and requested_generated is not None:
        multiplier = 2 if is_hdsi_s2c_run else 3
        expanded_pool = min(len(split_samples), max(requested_generated * multiplier, requested_generated))
        effective_max_samples = (
            max(int(max_samples or 0), expanded_pool) if max_samples is not None else expanded_pool
        )
    samples = _limit_samples_by_strategy(split_samples, effective_max_samples, strategy=sample_strategy)
    method_label = (
        "Spec-SRT-LGI"
        if preset_key == "spec-srt"
        else "HDSI-S2C-DE-LGI"
        if use_srt and use_ucs and use_hdsi and use_hdsi_s2c
        else
        "E-SRT+HDSI-PD-DE-LGI"
        if use_srt and use_ucs and use_e_srt and use_hdsi and use_hdsi_pd
        else "SRT+HDSI-PD-DE-LGI"
        if use_srt and use_ucs and use_hdsi and use_hdsi_pd
        else
        "E-SRT+HDSI-DE-LGI"
        if use_srt and use_ucs and use_e_srt and use_hdsi
        else "SRT+HDSI-DE-LGI"
        if use_srt and use_ucs and use_hdsi
        else "E-SRT-DE-LGI"
        if use_srt and use_ucs and (use_e_srt or use_seca)
        else "SRT-DE-LGI"
        if use_srt and use_ucs
        else "HDSI-DE-LGI"
        if use_ucs and use_hdsi
        else "DRR-DE-LGI"
        if use_drr and use_ucs
        else "UCS-DE-LGI"
        if use_ucs
        else "DE-LGI"
    )
    console.print(
        f"[cyan]{method_label} targets: {len(samples)}; split={generation_split}; "
        f"domain={effective_domain}; per_class={per_class}; candidates={max(1, int(candidates))}[/cyan]"
    )

    cfg = DELGIConfig(
        method_preset=preset_key,
        resolution=int(resolution),
        dilation_factor=float(dilation_factor),
        num_inference_steps=max(4, int(steps)),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        candidates_per_sample=max(1, int(candidates)),
        eta_min=float(eta_min),
        eta_max=float(eta_max),
        shell_strength=float(shell_strength),
        background_leakage=float(background_leakage),
        background_propagation_sigma=float(background_propagation_sigma),
        boundary_smoothing_sigma=float(boundary_smoothing_sigma),
        feather_radius=float(feather_radius),
        latent_energy_scale=float(latent_energy_scale),
        latent_energy_floor=float(latent_energy_floor),
        latent_energy_ceiling=float(latent_energy_ceiling),
        energy_jitter=float(energy_jitter),
        min_projection_scale=float(min_projection_scale),
        max_projection_scale=float(max_projection_scale),
        use_ucs=bool(use_ucs),
        spectrum_jitter=float(spectrum_jitter),
        utility_guidance=float(utility_guidance),
        spectrum_projection_strength=float(spectrum_projection_strength),
        spectrum_orientation_strength=float(spectrum_orientation_strength),
        core_renoise_strength=float(core_renoise_strength),
        diversity_weight=float(diversity_weight),
        use_drr=bool(use_drr),
        residual_bank_mix=float(residual_bank_mix),
        pseudo_suppression_strength=float(pseudo_suppression_strength),
        structure_jitter=float(structure_jitter),
        diversity_delta_target=float(diversity_delta_target),
        outside_delta_budget=float(outside_delta_budget),
        use_distant_spectrum=bool(use_distant_spectrum),
        distant_spectrum_candidates=max(2, int(distant_spectrum_candidates)),
        distant_spectrum_weight=float(distant_spectrum_weight),
        use_hdsi=bool(use_hdsi),
        hdsi_candidates=max(2, int(hdsi_candidates)),
        hdsi_hardness_weight=float(hdsi_hardness_weight),
        hdsi_validity_weight=float(hdsi_validity_weight),
        hdsi_diversity_weight=float(hdsi_diversity_weight),
        hdsi_projection_boost=float(hdsi_projection_boost),
        hdsi_tail_strength=float(hdsi_tail_strength),
        hdsi_min_validity_score=float(hdsi_min_validity_score),
        hdsi_min_source_distance=float(hdsi_min_source_distance),
        use_hdsi_s2c=bool(use_hdsi_s2c),
        hdsi_s2c_structure_weight=float(hdsi_s2c_structure_weight),
        hdsi_s2c_image_weight=float(hdsi_s2c_image_weight),
        hdsi_s2c_min_score=float(hdsi_s2c_min_score),
        hdsi_s2c_strict_gate=bool(hdsi_s2c_strict_gate),
        hdsi_s2c_spectrum_tolerance=float(hdsi_s2c_spectrum_tolerance),
        use_hdsi_pd=bool(use_hdsi_pd),
        hdsi_pd_prototype_count=max(1, int(hdsi_pd_prototype_count)),
        hdsi_pd_strength=float(hdsi_pd_strength),
        hdsi_pd_phase_strength=float(hdsi_pd_phase_strength),
        hdsi_pd_late_detail_strength=float(hdsi_pd_late_detail_strength),
        hdsi_pd_detail_start=float(hdsi_pd_detail_start),
        hdsi_pd_late_renoise_strength=float(hdsi_pd_late_renoise_strength),
        use_band_recomposition=bool(use_band_recomposition),
        band_donor_count=max(1, int(band_donor_count)),
        band_recomposition_strength=float(band_recomposition_strength),
        use_srt=bool(use_srt),
        srt_transport_strength=float(srt_transport_strength),
        srt_bbox_jitter=float(srt_bbox_jitter),
        srt_scale_jitter=float(srt_scale_jitter),
        srt_boundary_roughness=float(srt_boundary_roughness),
        srt_component_jitter=float(srt_component_jitter),
        srt_source_preservation=float(srt_source_preservation),
        srt_bbox_update=bool(srt_bbox_update),
        srt_regeneration_strength=float(srt_regeneration_strength),
        srt_regeneration_until=float(srt_regeneration_until),
        srt_late_texture_mix=float(srt_late_texture_mix),
        srt_s2i_visible_delta_target=float(srt_s2i_visible_delta_target),
        srt_s2i_min_ratio=float(srt_s2i_min_ratio),
        use_e_srt=bool(use_e_srt),
        e_srt_evidence_strength=float(e_srt_evidence_strength),
        e_srt_background_strength=float(e_srt_background_strength),
        e_srt_min_core_energy_ratio=float(e_srt_min_core_energy_ratio),
        use_seca=bool(use_seca),
        seca_min_coverage=float(seca_min_coverage),
        seca_max_leakage=float(seca_max_leakage),
        seca_min_focus=float(seca_min_focus),
        seca_strict=bool(seca_strict),
        srt_strict_s2i_gate=bool(srt_strict_s2i_gate),
    )

    sample_groups: Dict[str, List[Any]] = {}
    if per_class:
        for sample in samples:
            sample_groups.setdefault(sample.cls_name, []).append(sample)
    else:
        sample_groups["all"] = list(samples)

    reference_groups: Dict[str, List[Any]] = {}
    if per_class:
        for sample in split_samples:
            reference_groups.setdefault(sample.cls_name, []).append(sample)
    else:
        reference_groups["all"] = list(split_samples)

    generated_paths: List[Path] = []
    for cls_name, cls_samples in sample_groups.items():
        class_model_dir = model_dir / slugify_class_name(cls_name)
        effective_model_dir = class_model_dir if per_class and class_model_dir.exists() else model_dir
        refs = reference_groups.get(cls_name, cls_samples) if per_class else reference_groups["all"]
        console.print(
            f"[cyan]{method_label} group '{cls_name}': "
            f"{len(cls_samples)} sample(s), model={effective_model_dir}[/cyan]"
        )
        generator = DELGIGenerator(effective_model_dir)
        try:
            generated_paths.extend(
                generator.generate_de_lgi(
                    cls_samples,
                    run_dir,
                    domain=effective_domain,
                    reference_samples=refs,
                    config=cfg,
                )
            )
        finally:
            generator.release()

    score_path = run_dir / "artifacts" / "de_lgi" / "candidate_scores.jsonl"
    final_manifest = {
        "method": (
            "spec_srt_lgi"
            if preset_key == "spec-srt"
            else "hdsi_s2c_de_lgi"
            if use_srt and use_ucs and use_hdsi and use_hdsi_s2c
            else
            "e_srt_hdsi_pd_de_lgi"
            if use_srt and use_ucs and use_e_srt and use_hdsi and use_hdsi_pd
            else "srt_hdsi_pd_de_lgi"
            if use_srt and use_ucs and use_hdsi and use_hdsi_pd
            else
            "e_srt_hdsi_de_lgi"
            if use_srt and use_ucs and use_e_srt and use_hdsi
            else "srt_hdsi_de_lgi"
            if use_srt and use_ucs and use_hdsi
            else "e_srt_de_lgi"
            if use_srt and use_ucs and (use_e_srt or use_seca)
            else "srt_de_lgi"
            if use_srt and use_ucs
            else "hdsi_de_lgi"
            if use_ucs and use_hdsi
            else "drr_de_lgi"
            if use_drr and use_ucs
            else "ucs_de_lgi"
            if use_ucs
            else "de_lgi"
        ),
        "dataset_root": str(dataset_root.resolve()),
        "model_dir": str(model_dir.resolve()),
        "generation_split": generation_split,
        "sample_strategy": sample_strategy,
        "preset": preset_key,
        "domain": effective_domain,
        "per_class": bool(per_class),
        "generated_images": len(generated_paths),
        "target_generated_images": requested_generated,
        "run_dir": str(run_dir.resolve()),
        "image_dir": str((run_dir / "images").resolve()),
        "score_path": str(score_path.resolve()),
        "config": asdict(cfg),
        "innovation": (
            [
                "HDSI selects hard-but-valid source-distant class defect spectra before reverse diffusion",
                "Spec-SRT transports source defect residual evidence into a bbox-inherited phi' and validates it with one visibility-constrained projection gate",
            ]
            if preset_key == "spec-srt"
            else [
                "HDSI selects hard-but-valid and source-distant class defect spectra before reverse diffusion",
                "S2C projects each selected spectrum into bbox-constrained phi' structure fields",
                "Generated candidates are re-measured with image-side defect spectra and ranked by target-spectrum realization",
                "E-SRT boosts weak phi' evidence while HDSI-S2C gates spectrum, visible-change, and label-contract consistency",
                "HDSI-PD remains disabled by default so prototype detail is not a separate contribution",
            ]
            if is_hdsi_s2c_run
            else [
                "E-SRT boosts weak transported structure evidence while damping off-structure residual leakage during reverse diffusion",
                "HDSI selects hard-but-valid class spectra and source-distant spectrum targets before projection",
                "HDSI-PD injects real defect prototype phase/detail residuals and releases high-frequency detail only in late core steps",
                "Source-bbox label inheritance keeps the detector label contract stable while phi' controls intra-box structure",
                "Curated selected-score export writes a balanced 50-image YOLO dataset with per-candidate quality traces",
            ]
            if preset_key == "e-srt-hdsi-pd"
            else [
                "class-calibrated defect-energy prior mined from real defect boxes",
                "per-step latent residual projection with background leakage and boundary energy constraints",
                "utility-guided causal defect-spectrum projection over frequency, direction, polarity, roughness, connectivity, P_fail and P_syn proxies",
                "HDSI hard-but-valid target spectrum selection with lightweight frequency-tail intervention",
                "HDSI-PD spectrum-phase defect prototype injection with late core-only detail release",
                "diversity-aware cross-instance latent residual recomposition for non-copy defect structure",
                "source-distant target spectrum sampling and band-wise low/mid/high residual recomposition",
                "structure-residual transport with a transported phi field and source-bbox label inheritance by default",
                "E-SRT evidence-guided residual feedback for weak phi evidence and background leakage",
                "SECA structure-evidence consistent synthetic bbox/mask derivation",
            ]
        ),
    }
    if preset_key == "spec-srt":
        summary = _build_spec_srt_summary(run_dir, final_manifest)
        final_manifest["spec_srt_summary_path"] = str((run_dir / "spec_srt_summary.json").resolve())
        final_manifest["spec_srt_summary"] = {
            "candidate_rows": summary.get("candidate_rows", 0),
            "accepted_candidates": summary.get("accepted_candidates", 0),
            "selected_rows": summary.get("selected_rows", 0),
            "global_gate_failures": summary.get("global_gate_failures", {}),
            "class_selected_images": {
                cls_name: cls_summary.get("selected_images", 0)
                for cls_name, cls_summary in summary.get("classes", {}).items()
            },
        }
        if requested_generated is not None:
            selected_rows = _select_balanced_generated_rows(_load_jsonl(score_path), requested_generated)
            curated_manifest = _write_curated_generated_dataset(
                run_dir,
                selected_rows,
                dataset_root=dataset_root,
                target_count=requested_generated,
            )
            final_manifest["curated_generated_dataset"] = curated_manifest
    elif is_hdsi_s2c_run:
        summary = _build_hdsi_s2c_summary(run_dir, final_manifest)
        final_manifest["hdsi_s2c_summary_path"] = str((run_dir / "hdsi_s2c_summary.json").resolve())
        final_manifest["hdsi_s2c_summary"] = {
            "candidate_rows": summary.get("candidate_rows", 0),
            "accepted_candidates": summary.get("accepted_candidates", 0),
            "selected_rows": summary.get("selected_rows", 0),
            "gate_pass_rows": summary.get("gate_pass_rows", 0),
            "global_gate_failures": summary.get("global_gate_failures", {}),
            "global_rejection_reasons": summary.get("global_rejection_reasons", {}),
            "selected_quality_mean": summary.get("selected_quality_mean", {}),
            "csv_paths": summary.get("csv_paths", {}),
            "class_selected_images": {
                cls_name: cls_summary.get("selected_images", 0)
                for cls_name, cls_summary in summary.get("classes", {}).items()
            },
        }
        if requested_generated is not None:
            selected_rows = _select_balanced_generated_rows(_load_jsonl(score_path), requested_generated)
            curated_manifest = _write_curated_generated_dataset(
                run_dir,
                selected_rows,
                dataset_root=dataset_root,
                target_count=requested_generated,
                curated_slug="hdsi_s2c",
                selection_label="balanced-by-class-selected-hdsi-s2c-score",
            )
            final_manifest["curated_generated_dataset"] = curated_manifest
    elif preset_key == "e-srt-hdsi-pd":
        summary = _build_e_srt_hdsi_pd_summary(run_dir, final_manifest)
        final_manifest["e_srt_hdsi_pd_summary_path"] = str((run_dir / "e_srt_hdsi_pd_summary.json").resolve())
        final_manifest["e_srt_hdsi_pd_summary"] = {
            "candidate_rows": summary.get("candidate_rows", 0),
            "accepted_candidates": summary.get("accepted_candidates", 0),
            "selected_rows": summary.get("selected_rows", 0),
            "pd_enabled_rows": summary.get("pd_enabled_rows", 0),
            "gate_pass_rows": summary.get("gate_pass_rows", 0),
            "global_rejection_reasons": summary.get("global_rejection_reasons", {}),
            "selected_quality_mean": summary.get("selected_quality_mean", {}),
            "class_selected_images": {
                cls_name: cls_summary.get("selected_images", 0)
                for cls_name, cls_summary in summary.get("classes", {}).items()
            },
        }
        if requested_generated is not None:
            selected_rows = _select_balanced_generated_rows(_load_jsonl(score_path), requested_generated)
            curated_manifest = _write_curated_generated_dataset(
                run_dir,
                selected_rows,
                dataset_root=dataset_root,
                target_count=requested_generated,
                curated_slug="e_srt_hdsi_pd",
                selection_label="balanced-by-class-selected-e-srt-hdsi-pd-score",
            )
            final_manifest["curated_generated_dataset"] = curated_manifest
    (run_dir / "de_lgi_run_manifest.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "de_lgi_manifest.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if generated_paths and make_mixed_dataset:
        effective_manifest = split_manifest
        if effective_manifest is None or not Path(effective_manifest).exists():
            effective_manifest = _write_split_manifest_from_voc_layout(dataset_root, run_dir / "split_manifest.json")
        effective_orig_images = orig_images_dir or _default_images_dir_for_dataset(dataset_root)
        effective_orig_labels = orig_labels_dir or _default_labels_dir_for_dataset(dataset_root)
        if not effective_orig_labels.exists():
            console.print(
                f"[yellow]Label directory not found ({effective_orig_labels}); skipping mixed dataset creation.[/yellow]"
            )
        else:
            create_mixed_dataset(
                orig_images_dir=effective_orig_images,
                orig_labels_dir=effective_orig_labels,
                manifest_path=effective_manifest,
                new_images_dir=run_dir / "images",
                run_output_dir=run_dir / "mixed_dataset",
                seed=bundle.dataset.seed,
                score_path=score_path,
                max_generated=mixed_max_generated,
                per_source_limit=mixed_per_source_limit,
                per_class_limit=mixed_per_class_limit,
                quality_threshold=mixed_quality_threshold,
                max_outside_delta=mixed_max_outside_delta,
                selection_strategy="balanced-quality",
            )
            console.print(f"[green]Mixed dataset written to {(run_dir / 'mixed_dataset').resolve()}[/green]")
    console.print(f"[green]DE-LGI run written to {run_dir.resolve()}[/green]")


@app.command()
def generate(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    guidance_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    lora_path: Path = typer.Argument(..., exists=True),
    output_dir: Path = typer.Option(Path("outputs/generated"), file_okay=False),
    caption_file: Optional[Path] = typer.Option(None, help="Path to auto-generated captions JSON file (will auto-generate if not provided)"),
    priority_class: Optional[str] = typer.Option(None, help="Prioritize generating this defect class first (e.g., 'inclusion')"),
    max_samples: Optional[int] = typer.Option(None, help="Limit the number of samples to generate (for quick smoke tests)"),
    generation_split: str = typer.Option(
        "all",
        "--generation-split",
        help="Generate only this split from data/processed/split_manifest.json; use 'train' for leakage-free augmentation.",
    ),
    split_manifest: Path = typer.Option(
        _default_split_manifest(),
        "--split-manifest",
        exists=True,
        help="Split manifest used by --generation-split and mixed dataset creation.",
    ),
    balanced_max_samples: bool = typer.Option(
        False,
        "--balanced-max-samples",
        help="When --max-samples is set, select targets round-robin across defect classes.",
    ),
    model_name: str = typer.Option("openai/clip-vit-large-patch14", help="CLIP model for prompt selection"),
    mode: str = typer.Option(
        "drft-v2",
        help="Generation preset. Only drft-v2 is retained.",
        case_sensitive=False,
    ),
    use_bbox_mask: bool = typer.Option(True, help="Apply soft bbox mask blending to keep background intact"),
    skip_large_bbox: bool = typer.Option(
        True,
        help="Skip samples whose bbox covers too much area (prevents full-image inpaint artifacts)",
    ),
    skip_large_bbox_ratio: float = typer.Option(
        0.6,
        help="Skip when bbox_area/(image_area) >= this ratio (suggest 0.6~0.8)",
    ),
    log_file: Optional[Path] = typer.Option(None, help="日志文件路径 (默认: 每次生成独立 run_xxx/run.log)"),
    # Auto-training options
    auto_train: bool = typer.Option(False, "--auto-train", help="生成后自动训练 YOLO 模型"),
    yolo_model: str = typer.Option("yolov8.yaml", help="YOLO 模型配置"),
    make_mixed_dataset: bool = typer.Option(
        True,
        "--make-mixed-dataset/--no-mixed-dataset",
        help="Create a mixed real+generated dataset after generation.",
    ),
    yolo_train_mode: str = typer.Option("default", help="YOLO train mode for --auto-train, e.g. default or a custom research mode"),
    epochs: int = typer.Option(300, help="训练 epochs"),
    patience: int = typer.Option(50, help="早停 patience"),
    batch: int = typer.Option(16, help="Batch size"),
    imgsz: int = typer.Option(640, help="图像大小"),
    device: str = typer.Option("0", help="CUDA 设备 (0, 1, cpu)"),
    workers: int = typer.Option(8, "--workers", help="YOLO dataloader workers（Windows 建议 0）"),
    discriminator_weights: Optional[Path] = typer.Option(None, hidden=True),
    ip_adapter_scale: float = typer.Option(0.6, hidden=True),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducible generation (default: use config value)"),
    drft_candidates: int = typer.Option(
        2,
        "--drft-candidates",
        help="Number of residual-flow candidates per sample for mode=drft-v2.",
    ),
    drft_quality_threshold: float = typer.Option(
        0.0,
        "--drft-quality-threshold",
        help="Record whether selected DRFT candidates fall below this image-quality threshold.",
    ),
    drft_context_dilation: float = typer.Option(
        1.65,
        "--drft-context-dilation",
        help="Expanded context-window scale for DRFT-v2 generation.",
    ),
    drft_context_shell_weight: float = typer.Option(
        0.55,
        "--drft-context-shell-weight",
        help="Boundary-shell edit weight for DRFT-v2 generation.",
    ),
    drft_context_core_weight: float = typer.Option(
        1.0,
        "--drft-context-core-weight",
        help="Hard-core edit weight for DRFT-v2 generation.",
    ),
    drft_context_core_threshold: float = typer.Option(
        0.10,
        "--drft-context-core-threshold",
        help="Soft-mask threshold used to build hard core edit regions.",
    ),
    drft_residual_seed_gain: float = typer.Option(
        0.0,
        "--drft-residual-seed-gain",
        help="Class-aware signed-residual seed strength for DRFT-v2 generation.",
    ),
    drft_context_min_score: float = typer.Option(
        0.985,
        "--drft-context-min-score",
        help="Minimum context quality score used by defect-first selection.",
    ),
    drft_defect_first_selection: bool = typer.Option(
        True,
        "--drft-defect-first-selection/--no-drft-defect-first-selection",
        help="Select DRFT-v2 candidates by defect evidence after context gating.",
    ),
    drft_steps: int = typer.Option(
        18,
        "--drft-steps",
        help="Number of residual-flow integration steps for mode=drft-v2.",
    ),
    mixed_max_generated: Optional[int] = typer.Option(
        None,
        "--mixed-max-generated",
        help="Maximum generated images to copy into the mixed training split.",
    ),
    mixed_per_source_limit: int = typer.Option(
        0,
        "--mixed-per-source-limit",
        help="Maximum generated images kept per original source image; 0 keeps all.",
    ),
    mixed_per_class_limit: int = typer.Option(
        0,
        "--mixed-per-class-limit",
        help="Maximum generated images kept per generated defect class; 0 keeps all.",
    ),
    mixed_quality_threshold: Optional[float] = typer.Option(
        None,
        "--mixed-quality-threshold",
        help="Minimum candidate quality.total required before a generated image enters the mixed dataset.",
    ),
    mixed_min_canvas_drop: Optional[float] = typer.Option(
        None,
        "--mixed-min-canvas-drop",
        help="Minimum canvas.quality.defect_evidence_drop required for mixed dataset inclusion.",
    ),
    mixed_max_outside_delta: Optional[float] = typer.Option(
        None,
        "--mixed-max-outside-delta",
        help="Maximum quality.outside_delta allowed for mixed dataset inclusion.",
    ),
    mixed_selection_strategy: str = typer.Option(
        "random",
        "--mixed-selection-strategy",
        help="Generated-image selection strategy: random, quality, or balanced-quality.",
    ),
) -> None:
    """Generate images with the retained DRFT-v2 LoRA path using auto-generated captions.

    使用 --auto-train 选项可在生成后自动训练 YOLO 并在测试集上评估。
    使用 --seed 选项可确保生成结果的可复现性。
    """
    bundle = _ensure_bundle(ctx)
    dataset_cfg = replace(bundle.dataset, root=dataset_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_lc = mode.lower().replace("_", "-")
    valid_modes = {"drft-v2"}
    if mode_lc not in valid_modes:
        console.print(
            f"[red]Unknown generation mode '{mode}'. "
            f"Valid modes: {', '.join(sorted(valid_modes))}.[/red]"
        )
        raise typer.Exit(code=2)
    reference_samples = collect_dataset_images(dataset_root)
    console.print(f"[cyan]{mode_lc.upper()} image-level targets: {len(reference_samples)} image(s).[/cyan]")
    samples = list(reference_samples)
    run_dir = output_dir / f"run_{mode_lc}_{timestamp}"
    image_dir = run_dir / "images"
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: Dict[str, str] = {}

    effective_log = log_file or (run_dir / "run.log")
    _setup_logging(effective_log)

    split_lc = generation_split.lower().strip()
    if split_lc and split_lc != "all":
        split_stems = _load_split_stems(split_manifest, split_lc)
        samples = _filter_samples_by_stems(samples, split_stems)
        reference_samples = list(samples)
        console.print(
            f"[cyan]Generation split restricted to '{split_lc}': {len(samples)} sample(s); "
            f"reference bank uses the same split to avoid leakage.[/cyan]"
        )
        if not samples:
            raise typer.BadParameter(f"No samples matched generation split '{split_lc}' in {split_manifest}")

    # Sort samples to prioritize specific class if requested
    if priority_class:
        console.print(f"Prioritizing generation for class: {priority_class}")
        samples = sorted(samples, key=lambda s: (s.cls_name != priority_class, s.image_path.stem))

    # Retained DRFT-family paths should process every target.
    if skip_large_bbox:
        console.print(f"[cyan]{mode_lc} path: disabling large bbox skipping to process all samples.[/cyan]")
        skip_large_bbox = False

    # Skip samples with overly-large bbox (inpaint would degenerate into full-image redraw)
    skipped_large_bbox_stems: List[str] = []
    if use_bbox_mask and skip_large_bbox:
        from xml.etree import ElementTree as ET

        ratio_thr = float(skip_large_bbox_ratio)
        if ratio_thr <= 0:
            ratio_thr = 0.6
        if ratio_thr > 1.0:
            ratio_thr = 1.0

        kept: List[Any] = []
        for s in samples:
            # Default NEU-DET size is 200x200; prefer reading from XML for robustness
            w = h = 200
            try:
                tree = ET.parse(s.annotation_path)
                w = int(tree.findtext("size/width") or w)
                h = int(tree.findtext("size/height") or h)
            except Exception:
                pass
            xmin, ymin, xmax, ymax = s.bbox
            bbox_area = max(0, xmax - xmin) * max(0, ymax - ymin)
            img_area = max(1, w * h)
            ratio = bbox_area / img_area
            if ratio >= ratio_thr:
                skipped_large_bbox_stems.append(s.image_path.stem)
            else:
                kept.append(s)

        if skipped_large_bbox_stems:
            console.print(
                f"[yellow]Skipping {len(skipped_large_bbox_stems)} sample(s) with large bbox "
                f"(ratio >= {ratio_thr:.2f}), e.g. {skipped_large_bbox_stems[:5]}[/yellow]"
            )
        samples = kept

    if max_samples is not None:
        if balanced_max_samples:
            groups: Dict[str, List[Any]] = {}
            for sample in samples:
                groups.setdefault(sample.cls_name, []).append(sample)
            balanced: List[Any] = []
            class_names = sorted(groups)
            cursor = 0
            while len(balanced) < max_samples and any(groups.values()):
                cls_name = class_names[cursor % len(class_names)]
                if groups[cls_name]:
                    balanced.append(groups[cls_name].pop(0))
                cursor += 1
            samples = balanced
            console.print(f"Balanced generation subset across {len(class_names)} class(es).")
        else:
            samples = samples[:max_samples]
        console.print(f"Limiting generation to {len(samples)} sample(s) for this run.")

    cfg = replace(bundle.generation)

    # Override seed if provided via CLI
    if seed is not None:
        cfg.seed = seed
        console.print(f"[green]Using seed {seed} for reproducible generation[/green]")
    else:
        console.print(f"[cyan]Using default seed {cfg.seed} from config[/cyan]")

    cfg.num_inference_steps = max(4, int(drft_steps))
    cfg.denoising_strength = 1.0
    if not is_drft_lora_path(lora_path):
        console.print("[red]DRFT-family generation requires a DRFT-LoRA adapter.[/red]")
        raise typer.Exit(code=2)
    drft_meta = read_drft_lora_metadata(lora_path)
    cfg.base_model = str(drft_meta.get("model_id") or "runwayml/stable-diffusion-inpainting")
    console.print("[cyan]Mode DRFT-v2: adaptive counterfactual canvas + class-aware residual-field + timestep-gated LoRA[/cyan]")
    console.print("[cyan]  -> frozen inpainting U-Net + DRFT conditional LoRA attention processors[/cyan]")
    console.print(f"[cyan]  -> candidates per sample: {max(1, int(drft_candidates))}; steps: {max(4, int(drft_steps))}[/cyan]")

    generator = DRFTGenerator(cfg)
    lora_defaults = replace(bundle.lora)
    lora_metadata = _load_lora_metadata(lora_path, lora_defaults)
    lora_settings = {
        "rank": lora_metadata.get("rank"),
        "alpha": lora_metadata.get("alpha"),
        "dropout_rate": lora_metadata.get("dropout_rate"),
        "target_modules": lora_metadata.get("target_modules"),
        "model_id": lora_metadata.get("model_id"),
    }
    training_hparams = lora_metadata.get("training_hyperparameters", {})
    other_config = {
        "base_model": cfg.base_model,
        "generation_seed": cfg.seed,
        "training_resolution": lora_metadata.get("resolution"),
        "training_seed": lora_metadata.get("seed"),
        "prompt_template": lora_metadata.get("prompt_template"),
        "mixed_precision": lora_metadata.get("mixed_precision"),
    }
    token_map = {cls: f"<neu_{cls}>" for cls in set(s.cls_name for s in samples)}

    # Always use auto-generated captions
    if not caption_file:
        caption_file = output_dir.parent / "captions.json"

    if not caption_file.exists():
        console.print(f"Caption file not found. Generating captions automatically...")
        # 默认使用 CLIP，但可以通过环境变量或配置启用 BLIP-2
        use_blip2 = False  # 可以通过参数添加
        if use_blip2:
            captions = generate_captions_with_blip2(
                samples=samples,
                token_map=token_map,
                output_file=caption_file,
                use_paper_keywords=True,
                lora_weight=1.0,
            )
        else:
            caption_generator = CaptionGenerator(model_name=model_name)
            captions = caption_generator.generate_with_token(samples, token_map, output_file=caption_file)
            caption_generator.cleanup()
        console.print(f"Generated {len(captions)} captions, saved to {caption_file}")
    else:
        console.print(f"Loading captions from {caption_file}")
        captions = load_captions_from_file(caption_file)

    # Snapshot artifacts (captions, config, LoRA metadata) for reproducibility
    if caption_file.exists():
        caption_dest = artifacts_dir / caption_file.name
        try:
            if caption_dest.resolve() != caption_file.resolve():
                shutil.copy2(caption_file, caption_dest)
        except FileNotFoundError:
            caption_dest = caption_file
        artifact_paths["captions"] = str(caption_dest.resolve())

    if bundle.source_path and bundle.source_path.exists():
        config_dest = artifacts_dir / bundle.source_path.name
        if config_dest.resolve() != bundle.source_path.resolve():
            shutil.copy2(bundle.source_path, config_dest)
        artifact_paths["config"] = str(config_dest.resolve())

    lora_config_path = lora_path.parent / "lora_config.json"
    if lora_config_path.exists():
        lora_config_dest = artifacts_dir / lora_config_path.name
        if lora_config_dest.resolve() != lora_config_path.resolve():
            shutil.copy2(lora_config_path, lora_config_dest)
        artifact_paths["lora_config"] = str(lora_config_dest.resolve())
    drft_artifact_name = "drft_v2_lora"
    artifact_paths["drft_lora_dir"] = str((artifacts_dir / drft_artifact_name).resolve())
    artifact_paths["drft_lora_candidate_scores"] = str((artifacts_dir / drft_artifact_name / "candidate_scores.jsonl").resolve())

    # Build prompt items using auto-generated captions.
    # Always append the class token to ensure it is actually used in the prompt.
    prompt_items = []
    cls_map = {}  # For local mode: stem -> class_name mapping
    for s in samples:
        target_key = _sample_target_key(s)
        token = token_map[s.cls_name]
        base_prompt = captions.get(s.image_path.stem, f"macro shot of {token} steel surface")
        prompt_items.append((target_key, f"{base_prompt}, {token}"))
        cls_map[target_key] = s.cls_name

    # DRFT-v2 does not use external conditioning maps.
    conditioning: Dict[str, Dict[str, Path]] = {}

    init_images = {_sample_target_key(s): s.image_path for s in samples}
    bbox_map = {_sample_target_key(s): s.bbox for s in samples}

    prompt_items_with_cls = [
        (_sample_target_key(s), f"{captions.get(s.image_path.stem, f'macro shot of {token_map[s.cls_name]} steel surface')}, {token_map[s.cls_name]}", s.cls_name)
        for s in samples
    ]
    generated_paths = generator.generate_drft_lora_inpaint(
        lora_path,
        prompt_items_with_cls,
        conditioning,
        image_dir,
        init_images=init_images,
        bbox_map=bbox_map,
        samples=samples,
        reference_samples=reference_samples,
        candidates_per_sample=drft_candidates,
        quality_threshold=drft_quality_threshold,
        variant=mode_lc,
        context_dilation=drft_context_dilation,
        context_shell_weight=drft_context_shell_weight,
        context_core_edit_weight=drft_context_core_weight,
        context_core_threshold=drft_context_core_threshold,
        residual_seed_gain=drft_residual_seed_gain,
        context_min_score=drft_context_min_score,
        defect_first_selection=drft_defect_first_selection,
    )

    manifest: Dict[str, object]
    if generated_paths:
        metrics_dest = run_dir / "metrics.json"
        evaluator = GenerationMetricsEvaluator()
        # When bbox-mask inpaint is enabled, evaluate metrics on bbox region only
        metric_bbox_map = bbox_map if use_bbox_mask else None
        metric_result = evaluator.evaluate(
            generated_paths,
            init_images,
            conditioning,
            bbox_map=metric_bbox_map,
            bbox_source_size=(200, 200),
        )
        run_details = {
            "timestamp": timestamp,
            "generated_images": len(generated_paths),
            "dataset_root": str(dataset_root.resolve()),
            "guidance_dir": str(guidance_dir.resolve()),
            "lora_path": str(lora_path.resolve()),
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "denoising_strength": cfg.denoising_strength,
            # Ensure JSON-serializable device field
            "device": str(evaluator.device),
            "output_directory": str(image_dir.resolve()),
            "max_samples": max_samples,
            "balanced_max_samples": bool(balanced_max_samples),
            "skip_large_bbox": bool(skip_large_bbox),
            "skip_large_bbox_ratio": float(skip_large_bbox_ratio),
            "skipped_large_bbox_count": len(skipped_large_bbox_stems),
            "skipped_large_bbox_stems": skipped_large_bbox_stems[:50],
            "drft_candidates": max(1, int(drft_candidates)),
            "drft_steps": max(3, int(drft_steps)),
            "generation_mode": mode_lc,
        }
        save_metrics(metric_result, metrics_dest, extra=run_details)
        artifact_paths["metrics"] = str(metrics_dest.resolve())

        manifest = {
            **run_details,
            "log_file": str(effective_log.resolve()) if effective_log else None,
            "metrics_file": str(metrics_dest.resolve()),
            "caption_file": str(caption_file.resolve()) if caption_file else None,
            "generated_files": [str(path.resolve()) for path in generated_paths],
            "lora_settings": lora_settings,
            "training_hyperparameters": training_hparams,
            "other_config": other_config,
            "artifacts": artifact_paths,
        }
    else:
        console.print("[yellow]未生成任何图像，跳过指标评估。[/yellow]")
        manifest = {
            "timestamp": timestamp,
            "generated_images": 0,
            "dataset_root": str(dataset_root.resolve()),
            "guidance_dir": str(guidance_dir.resolve()),
            "lora_path": str(lora_path.resolve()),
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "denoising_strength": cfg.denoising_strength,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "output_directory": str(image_dir.resolve()),
            "max_samples": max_samples,
            "balanced_max_samples": bool(balanced_max_samples),
            "skip_large_bbox": bool(skip_large_bbox),
            "skip_large_bbox_ratio": float(skip_large_bbox_ratio),
            "skipped_large_bbox_count": len(skipped_large_bbox_stems),
            "skipped_large_bbox_stems": skipped_large_bbox_stems[:50],
            "drft_candidates": max(1, int(drft_candidates)),
            "drft_steps": max(3, int(drft_steps)),
            "generation_mode": mode_lc,
            "log_file": str(effective_log.resolve()) if effective_log else None,
            "metrics_file": None,
            "caption_file": str(caption_file.resolve()) if caption_file else None,
            "generated_files": [],
            "lora_settings": lora_settings,
            "training_hyperparameters": training_hparams,
            "other_config": other_config,
            "artifacts": artifact_paths,
        }

    manifest_path = run_dir / "run_context.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Automatically create mixed dataset after image generation
    if generated_paths and make_mixed_dataset:
        try:
            console.print("[cyan]自动生成混合数据集...[/cyan]")
            # Try to infer paths from dataset_root and output_dir
            # First, try to find the project root
            project_root = Path(__file__).resolve().parent.parent.parent  # src/neu_det_pipeline -> project root

            # Try to find original images directory
            orig_images_dir = dataset_root / "IMAGES"
            if not orig_images_dir.exists():
                # Try project root / data / raw / NEU-DET / IMAGES
                orig_images_dir = project_root / "data" / "raw" / "NEU-DET" / "IMAGES"

            # Try to find labels directory (新目录结构)
            orig_labels_dir = project_root / "data" / "processed" / "yolo" / "labels"
            if not orig_labels_dir.exists():
                # Fallback: Try old path
                orig_labels_dir = output_dir.parent / "yolo_baseline" / "labels"
                if not orig_labels_dir.exists():
                    orig_labels_dir = project_root / "outputs" / "yolo_baseline" / "labels"

            # Try to find manifest file
            manifest_path = Path(split_manifest)
            if not manifest_path.exists():
                manifest_path = project_root / "data" / "processed" / "split_manifest.json"
            if not manifest_path.exists():
                # Fallback: Try old paths
                manifest_path = output_dir.parent / "split_manifest.json"
                if not manifest_path.exists():
                    manifest_path = project_root / "outputs" / "split_manifest.json"

            console.print(f"[dim]标签目录: {orig_labels_dir}[/dim]")
            console.print(f"[dim]Manifest: {manifest_path}[/dim]")

            # new_images_dir is the image_dir (run_dir / "images")
            new_images_dir = image_dir
            # run_output_dir will be set to run_dir / "mixed_dataset" by default
            run_output_dir = run_dir / "mixed_dataset"
            mixed_score_path: Optional[Path] = artifacts_dir / drft_artifact_name / "candidate_scores.jsonl"
            if mixed_score_path is not None and not mixed_score_path.exists():
                mixed_score_path = None

            # Only proceed if manifest exists
            if manifest_path.exists():
                effective_mixed_strategy = mixed_selection_strategy
                create_mixed_dataset(
                    orig_images_dir=orig_images_dir,
                    orig_labels_dir=orig_labels_dir,
                    manifest_path=manifest_path,
                    new_images_dir=new_images_dir,
                    run_output_dir=run_output_dir,
                    seed=dataset_cfg.seed,
                    score_path=mixed_score_path,
                    max_generated=mixed_max_generated,
                    per_source_limit=mixed_per_source_limit,
                    per_class_limit=mixed_per_class_limit,
                    quality_threshold=mixed_quality_threshold,
                    min_canvas_drop=mixed_min_canvas_drop,
                    max_outside_delta=mixed_max_outside_delta,
                    selection_strategy=effective_mixed_strategy,
                )
                console.print(f"[green]混合数据集已生成: {run_output_dir}[/green]")
            else:
                console.print(f"[yellow]未找到 split_manifest.json ({manifest_path})，跳过混合数据集生成。[/yellow]")
                console.print(f"[yellow]请确保 split_manifest.json 存在于 outputs 目录下。[/yellow]")
        except Exception as exc:  # noqa: BLE001 - we want to log and continue
            console.print(f"[yellow]混合数据集生成失败: {exc}。请手动检查。[/yellow]")

    # ═══════════════════════════════════════════════════════════════
    # 自动训练: 如果启用 --auto-train，在混合数据集上训练 YOLO
    # ═══════════════════════════════════════════════════════════════
    if generated_paths and not make_mixed_dataset:
        console.print("[cyan]Skipping mixed dataset creation (--no-mixed-dataset).[/cyan]")

    if auto_train and generated_paths:
        mixed_dataset_dir = run_dir / "mixed_dataset"
        data_yaml_path = mixed_dataset_dir / "data.yaml"

        if data_yaml_path.exists():
            console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
            console.print("[bold cyan]              自动训练: YOLO 目标检测[/bold cyan]")
            console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

            try:
                import gc

                # 实验结果放在生成目录下
                experiment_name = "yolo_training"
                experiment_dir = run_dir / experiment_name
                experiment_dir.mkdir(parents=True, exist_ok=True)

                console.print(f"[cyan]数据集: {data_yaml_path}[/cyan]")
                console.print(f"[cyan]模型: {yolo_model}[/cyan]")
                console.print(f"[cyan]Train mode: {yolo_train_mode}[/cyan]")
                console.print(f"[cyan]参数: epochs={epochs}, patience={patience}, batch={batch}, imgsz={imgsz}[/cyan]")
                console.print(f"[cyan]输出: {experiment_dir}[/cyan]\n")

                # 清理 GPU 内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # 初始化并训练模型
                # 使用绝对路径字符串避免 Windows 路径问题
                data_yaml_str = str(data_yaml_path.resolve()).replace("\\", "/")
                project_str = str(run_dir.resolve()).replace("\\", "/")

                YOLOBackend, trainer_cls = _resolve_yolo_backend(yolo_train_mode, yolo_model, data_yaml_path)
                yolo = YOLOBackend(yolo_model)
                train_kwargs = {
                    "data": data_yaml_str,
                    "epochs": epochs,
                    "patience": patience,
                    "batch": batch,
                    "imgsz": imgsz,
                    "project": project_str,
                    "name": experiment_name,
                    "exist_ok": True,
                    "device": device,
                    "verbose": True,
                    "workers": workers,
                }
                if Path(yolo_model).suffix.lower() in {".yaml", ".yml"}:
                    train_kwargs["pretrained"] = False
                if trainer_cls is not None:
                    train_kwargs["trainer"] = trainer_cls
                train_results = yolo.train(**train_kwargs)

                best_model_path = experiment_dir / "weights" / "best.pt"
                console.print(f"\n[bold green]✓ 训练完成![/bold green]")
                console.print(f"[green]最佳模型: {best_model_path}[/green]")

                # 在测试集上评估
                console.print("\n[yellow]在测试集上评估...[/yellow]\n")
                test_model = YOLOBackend(str(best_model_path))
                test_metrics = test_model.val(
                    data=str(data_yaml_path),
                    split="test",
                    project=str(experiment_dir),
                    name="test_results",
                    exist_ok=True,
                    verbose=True,
                    workers=workers,
                )

                # 提取并显示关键指标
                test_results = {
                    "mAP50": float(test_metrics.box.map50) if hasattr(test_metrics.box, 'map50') else None,
                    "mAP50-95": float(test_metrics.box.map) if hasattr(test_metrics.box, 'map') else None,
                    "precision": float(test_metrics.box.mp) if hasattr(test_metrics.box, 'mp') else None,
                    "recall": float(test_metrics.box.mr) if hasattr(test_metrics.box, 'mr') else None,
                }

                # 计算 F1
                if test_results["precision"] and test_results["recall"]:
                    p, r = test_results["precision"], test_results["recall"]
                    test_results["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                else:
                    test_results["f1"] = None

                # 按类别的 AP
                if hasattr(test_metrics.box, 'ap50') and test_metrics.box.ap50 is not None:
                    test_results["per_class_ap50"] = {
                        name: float(ap) for name, ap in zip(test_metrics.names.values(), test_metrics.box.ap50)
                    }

                console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
                console.print("[bold cyan]              测试集评估结果[/bold cyan]")
                console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

                console.print(f"[green]mAP@50:      {test_results['mAP50']:.4f}[/green]" if test_results['mAP50'] else "[yellow]mAP@50: N/A[/yellow]")
                console.print(f"[green]mAP@50-95:   {test_results['mAP50-95']:.4f}[/green]" if test_results['mAP50-95'] else "[yellow]mAP@50-95: N/A[/yellow]")
                console.print(f"[green]Precision:   {test_results['precision']:.4f}[/green]" if test_results['precision'] else "[yellow]Precision: N/A[/yellow]")
                console.print(f"[green]Recall:      {test_results['recall']:.4f}[/green]" if test_results['recall'] else "[yellow]Recall: N/A[/yellow]")
                console.print(f"[green]F1 Score:    {test_results['f1']:.4f}[/green]" if test_results['f1'] else "[yellow]F1: N/A[/yellow]")

                if "per_class_ap50" in test_results:
                    console.print("\n[cyan]每类 AP@50:[/cyan]")
                    for cls_name, ap in test_results["per_class_ap50"].items():
                        console.print(f"  {cls_name}: {ap:.4f}")

                # 保存完整实验报告
                experiment_report = {
                    "experiment": {
                        "timestamp": timestamp,
                        "generation_run": str(run_dir.resolve()),
                        "data_yaml": str(data_yaml_path.resolve()),
                        "model": yolo_model,
                        "train_mode": yolo_train_mode,
                        "hyperparameters": {
                            "epochs": epochs,
                            "patience": patience,
                            "batch": batch,
                            "imgsz": imgsz,
                        },
                    },
                    "generation_metrics": manifest.get("metrics_file") if isinstance(manifest, dict) else None,
                    "test_metrics": test_results,
                    "best_model": str(best_model_path.resolve()),
                    "output_dir": str(experiment_dir.resolve()),
                }

                report_path = experiment_dir / "experiment_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(experiment_report, indent=2, ensure_ascii=False))

                console.print(f"\n[green]实验报告已保存: {report_path}[/green]")
                console.print(f"\n[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
                console.print(f"[bold magenta]  完整流水线完成![/bold magenta]")
                console.print(f"[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]\n")
                console.print(f"[green]输出目录: {run_dir}[/green]")
                console.print(f"[green]训练结果: {experiment_dir}[/green]")
                console.print(f"[green]最佳模型: {best_model_path}[/green]\n")

            except ImportError:
                console.print("[red]错误: 未安装 ultralytics，请运行: pip install ultralytics[/red]")
            except Exception as exc:
                console.print(f"[red]自动训练失败: {exc}[/red]")
                import traceback
                console.print(f"[red]{traceback.format_exc()}[/red]")
        else:
            console.print(f"[yellow]未找到混合数据集 {data_yaml_path}，跳过自动训练。[/yellow]")



@app.command("train-yolo")
def train_yolo(
    ctx: typer.Context,
    data_yaml: Path = typer.Argument(..., exists=True, help="YOLO 数据集配置文件 (data.yaml)"),
    model: str = typer.Option("yolov8.yaml", help="YOLO 模型配置 (yolov8.yaml/yolov8s.yaml/yolov8n.pt)"),
    train_mode: str = typer.Option("default", help="Training mode, e.g. default or a custom research mode"),
    epochs: int = typer.Option(1000, help="训练 epochs"),
    patience: int = typer.Option(50, help="早停 patience"),
    batch: int = typer.Option(8, help="Batch size"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    imgsz: int = typer.Option(640, help="图像大小"),
    project: Path = typer.Option(Path("outputs/experiments"), help="实验输出目录"),
    name: Optional[str] = typer.Option(None, help="实验名称 (默认: 自动生成)"),
    test_after_train: bool = typer.Option(True, help="训练后自动在测试集上评估"),
    device: str = typer.Option("0", help="CUDA 设备 (0, 1, cpu)"),
    workers: int = typer.Option(0, "--workers", help="YOLO dataloader workers（Windows 建议 0）"),
) -> None:
    """
    使用 YOLO 训练目标检测模型并在测试集上评估。

    示例:
        neu-det train-yolo outputs/generated/run_xxx/mixed_dataset/data.yaml
        neu-det train-yolo data.yaml --epochs 500 --batch 16
    """
    # 生成实验名称
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"train_{timestamp}"

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]              YOLO 目标检测训练[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    console.print(f"[cyan]数据集: {data_yaml}[/cyan]")
    console.print(f"[cyan]模型: {model}[/cyan]")
    console.print(f"[cyan]Train mode: {train_mode}[/cyan]")
    console.print(f"[cyan]参数: epochs={epochs}, patience={patience}, batch={batch}, imgsz={imgsz}, seed={seed}[/cyan]")
    console.print(f"[cyan]请求输出: {project / name}[/cyan]\n")

    try:
        YOLOBackend, trainer_cls = _resolve_yolo_backend(train_mode, model, data_yaml)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to initialize YOLO backend for train_mode={train_mode}: {exc}[/red]")
        raise typer.Exit(1)

    # 初始化模型
    yolo_model = YOLOBackend(model)

    # 训练
    console.print("[yellow]开始训练...[/yellow]\n")
    yolo_model = YOLOBackend(model)
    train_kwargs = {
        "data": str(data_yaml.resolve()),
        "epochs": epochs,
        "patience": patience,
        "batch": batch,
        "imgsz": imgsz,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "device": device,
        "verbose": True,
        "workers": workers,
        "seed": seed,
    }
    if Path(model).suffix.lower() in {".yaml", ".yml"}:
        train_kwargs["pretrained"] = False
    if trainer_cls is not None:
        train_kwargs["trainer"] = trainer_cls

    results = yolo_model.train(**train_kwargs)

    save_dir_raw = getattr(results, "save_dir", None) or getattr(getattr(yolo_model, "trainer", None), "save_dir", None)
    save_dir = Path(save_dir_raw).resolve() if save_dir_raw else (project / name).resolve()
    weights_dir = save_dir / "weights"
    best_model_path = weights_dir / "best.pt"
    if not best_model_path.exists():
        last_model_path = weights_dir / "last.pt"
        if last_model_path.exists():
            best_model_path = last_model_path
        else:
            raise FileNotFoundError(f"Could not find trained weights under {weights_dir}")

    console.print(f"\n[bold green]✓ 训练完成![/bold green]")
    console.print(f"[green]保存目录: {save_dir}[/green]")
    console.print(f"[green]最佳模型: {best_model_path}[/green]")

    # 训练后在测试集上评估
    if test_after_train:
        console.print("\n[yellow]在测试集上评估...[/yellow]\n")
        test_results = _evaluate_on_test(best_model_path, data_yaml, save_dir, device=device, workers=workers, train_mode=train_mode)

        # 保存完整指标报告
        _save_experiment_report(
            save_dir,
            data_yaml,
            model,
            train_mode,
            epochs,
            patience,
            batch,
            imgsz,
            seed,
            results,
            test_results,
        )


def _evaluate_on_test(
    model_path: Path,
    data_yaml: Path,
    output_dir: Path,
    *,
    device: str = "0",
    workers: int = 0,
    train_mode: str = "default",
) -> Dict[str, Any]:
    """在测试集上评估模型并返回指标。"""
    YOLOBackend, _trainer_cls = _resolve_yolo_backend(train_mode, str(model_path), data_yaml)

    model = YOLOBackend(str(model_path))

    # 在测试集上验证
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=str(output_dir),
        name="test_results",
        exist_ok=True,
        verbose=True,
        device=device,
        workers=workers,
    )

    # 提取关键指标
    results = {
        "mAP50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else None,
        "mAP50-95": float(metrics.box.map) if hasattr(metrics.box, 'map') else None,
        "precision": float(metrics.box.mp) if hasattr(metrics.box, 'mp') else None,
        "recall": float(metrics.box.mr) if hasattr(metrics.box, 'mr') else None,
        "f1": None,  # 计算 F1
    }

    # 计算 F1
    if results["precision"] and results["recall"]:
        p, r = results["precision"], results["recall"]
        results["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # 按类别的指标
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        results["per_class_ap50"] = {
            name: float(ap) for name, ap in zip(metrics.names.values(), metrics.box.ap50)
        }

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]              测试集评估结果[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    console.print(f"[green]mAP@50:      {results['mAP50']:.4f}[/green]" if results['mAP50'] else "[yellow]mAP@50: N/A[/yellow]")
    console.print(f"[green]mAP@50-95:   {results['mAP50-95']:.4f}[/green]" if results['mAP50-95'] else "[yellow]mAP@50-95: N/A[/yellow]")
    console.print(f"[green]Precision:   {results['precision']:.4f}[/green]" if results['precision'] else "[yellow]Precision: N/A[/yellow]")
    console.print(f"[green]Recall:      {results['recall']:.4f}[/green]" if results['recall'] else "[yellow]Recall: N/A[/yellow]")
    console.print(f"[green]F1 Score:    {results['f1']:.4f}[/green]" if results['f1'] else "[yellow]F1: N/A[/yellow]")

    if "per_class_ap50" in results:
        console.print("\n[cyan]每类 AP@50:[/cyan]")
        for cls_name, ap in results["per_class_ap50"].items():
            console.print(f"  {cls_name}: {ap:.4f}")

    return results


def _save_experiment_report(
    output_dir: Path,
    data_yaml: Path,
    model: str,
    train_mode: str,
    epochs: int,
    patience: int,
    batch: int,
    imgsz: int,
    seed: int,
    train_results: Any,
    test_results: Dict[str, Any],
) -> None:
    """保存实验报告到 JSON 文件。"""
    report = {
        "experiment": {
            "timestamp": datetime.now().isoformat(),
            "data_yaml": str(data_yaml.resolve()),
            "model": model,
            "train_mode": train_mode,
            "hyperparameters": {
                "epochs": epochs,
                "patience": patience,
                "batch": batch,
                "imgsz": imgsz,
                "seed": seed,
            },
        },
        "test_metrics": test_results,
        "output_dir": str(output_dir.resolve()),
    }

    report_path = output_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    console.print(f"\n[green]实验报告已保存: {report_path}[/green]")


@app.command("full-pipeline")
def full_pipeline(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False, help="NEU-DET 数据集根目录"),
    guidance_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="引导特征目录"),
    lora_path: Path = typer.Argument(..., exists=True, help="LoRA 权重路径"),
    output_dir: Path = typer.Option(Path("outputs/generated"), help="生成输出目录"),
    # Generation options
    max_samples: Optional[int] = typer.Option(None, help="限制生成样本数量"),
    mode: str = typer.Option("drft-v2", help="Generation mode. Only drft-v2 is retained."),
    # Training options
    yolo_model: str = typer.Option("yolov8.yaml", help="YOLO 模型配置"),
    epochs: int = typer.Option(1000, help="训练 epochs"),
    patience: int = typer.Option(50, help="早停 patience"),
    batch: int = typer.Option(8, help="Batch size"),
    imgsz: int = typer.Option(640, help="图像大小"),
    device: str = typer.Option("0", help="CUDA 设备"),
) -> None:
    """
    完整流水线: 生成合成图像 → 创建混合数据集 → YOLO训练 → 测试集评估

    一键运行完整的数据增强和训练流程。

    示例:
        neu-det full-pipeline data/raw/NEU-DET outputs/guidance models/lora/lora.safetensors
        neu-det full-pipeline data/raw/NEU-DET outputs/guidance models/lora/lora.safetensors --max-samples 100 --epochs 500
    """
    console.print("\n[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]           NEU-DET 完整流水线[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]\n")

    console.print("[bold]阶段 1/3: 生成合成图像[/bold]")
    console.print("─" * 60)

    # Step 1: Generate images (调用现有的 generate 命令逻辑)
    bundle = _ensure_bundle(ctx)
    dataset_cfg = replace(bundle.dataset, root=dataset_root)
    samples = collect_dataset(dataset_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    image_dir = run_dir / "images"
    run_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # 简化的生成逻辑 - 调用 generate 命令的核心部分
    # 这里我们直接调用 generate 命令
    ctx.invoke(
        generate,
        dataset_root=dataset_root,
        guidance_dir=guidance_dir,
        lora_path=lora_path,
        output_dir=output_dir,
        max_samples=max_samples,
        mode=mode,
    )

    # 查找最新生成的 run 目录
    run_dirs = sorted(output_dir.glob("run_*"), key=lambda x: x.name, reverse=True)
    if not run_dirs:
        console.print("[red]错误: 未找到生成的 run 目录[/red]")
        raise typer.Exit(1)

    latest_run = run_dirs[0]
    mixed_dataset_dir = latest_run / "mixed_dataset"
    data_yaml_path = mixed_dataset_dir / "data.yaml"

    if not data_yaml_path.exists():
        console.print(f"[red]错误: 混合数据集未生成: {data_yaml_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[green]✓ 合成图像已生成: {latest_run}[/green]")
    console.print(f"[green]✓ 混合数据集: {mixed_dataset_dir}[/green]\n")

    # Step 2: Train YOLO
    console.print("[bold]阶段 2/3: YOLO 训练[/bold]")
    console.print("─" * 60)

    experiment_name = f"exp_{timestamp}"
    project_dir = Path("outputs/experiments")

    ctx.invoke(
        train_yolo,
        data_yaml=data_yaml_path,
        model=yolo_model,
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        project=project_dir,
        name=experiment_name,
        test_after_train=True,
        device=device,
    )

    console.print("\n[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]           流水线完成![/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]\n")

    console.print(f"[green]生成目录: {latest_run}[/green]")
    console.print(f"[green]实验目录: {project_dir / experiment_name}[/green]")
    console.print(f"[green]最佳模型: {project_dir / experiment_name / 'weights' / 'best.pt'}[/green]")
    console.print(f"[green]实验报告: {project_dir / experiment_name / 'experiment_report.json'}[/green]\n")


if __name__ == "__main__":
    app()
