from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from rich.progress import track

from ..data.loader import DefectSample
from ..guidance.drft import (
    AnalyticResidualFlowSampler,
    DRFTResidualPrototypeBank,
    append_jsonl,
    build_adaptive_counterfactual_canvas,
    build_class_aware_defect_residual_field,
    build_context_preserved_drft_contract,
    save_drft_artifacts,
    score_drft_candidate,
    score_drft_context_contract,
)
from ..guidance.mask import create_composite_preserving_background
from ..guidance.morphology import MorphologyPrior, build_morphology_calibrated_mask, scale_bbox


@dataclass(frozen=True)
class ContextSGDAConfig:
    target_size: tuple[int, int] = (512, 512)
    steps: int = 18
    candidates_per_sample: int = 2
    seed: int = 42
    context_dilation: float = 1.65
    context_shell_weight: float = 0.55
    context_core_weight: float = 1.0
    context_core_threshold: float = 0.10
    feather_radius: float = 1.4


def _sample_target_key(sample: DefectSample) -> str:
    return str(getattr(sample, "target_key", sample.image_path.stem))


def _instance_protect_bboxes(
    sample: DefectSample,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    objects = tuple(getattr(sample, "objects", ()) or ())
    if len(objects) <= 1:
        return []
    current_index = int(getattr(sample, "object_index", 0))
    boxes: list[tuple[int, int, int, int]] = []
    for obj in objects:
        if int(getattr(obj, "index", -1)) == current_index:
            continue
        boxes.append(scale_bbox(obj.bbox, source_size, target_size))
    return boxes


def _build_instance_protect_mask(
    sample: DefectSample,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> Image.Image | None:
    boxes = _instance_protect_bboxes(sample, source_size, target_size)
    if not boxes:
        return None
    mask = Image.new("L", target_size, 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(box, fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(5))
    target_box = scale_bbox(sample.bbox, source_size, target_size)
    ImageDraw.Draw(mask).rectangle(target_box, fill=0)
    return mask if np.asarray(mask).max() > 0 else None


def _subtract_protect_mask(mask: Image.Image, protect_mask: Image.Image | None) -> Image.Image:
    if protect_mask is None:
        return mask
    arr = np.asarray(mask.convert("L"), dtype=np.uint8).copy()
    protect = np.asarray(protect_mask.resize(mask.size, Image.Resampling.NEAREST).convert("L"), dtype=np.uint8)
    arr[protect > 0] = 0
    return Image.fromarray(arr, mode="L")


def _mean_abs_residual(field: Any) -> float:
    residual = np.asarray(field.signed_residual, dtype=np.float32)
    core = np.asarray(field.soft_mask, dtype=np.float32) > 0.08
    if not core.any():
        return 0.0
    return float(np.mean(np.abs(residual[core])))


def _residual_concentration(field: Any, edit_mask: Image.Image) -> float:
    residual = np.abs(np.asarray(field.signed_residual, dtype=np.float32))
    edit = np.asarray(
        edit_mask.resize((residual.shape[1], residual.shape[0]), Image.Resampling.BILINEAR).convert("L"),
        dtype=np.float32,
    ) / 255.0
    total = float(np.sum(residual) + 1e-6)
    inside = float(np.sum(residual * np.clip(edit, 0.0, 1.0)))
    return float(max(0.0, min(1.0, inside / total)))


def _selection_score(
    quality: Any,
    context_quality: Any,
    field: Any,
    edit_mask: Image.Image,
) -> tuple[float, dict[str, float]]:
    residual_strength = 1.0 - math.exp(-_mean_abs_residual(field) / 18.0)
    concentration = _residual_concentration(field, edit_mask)
    score = (
        0.38 * float(quality.total)
        + 0.34 * float(context_quality.total)
        + 0.18 * float(residual_strength)
        + 0.10 * float(concentration)
    )
    return float(score), {
        "quality_total": float(quality.total),
        "context_total": float(context_quality.total),
        "residual_strength": float(residual_strength),
        "residual_concentration": float(concentration),
    }


class ContextSGDAGenerator:
    """Context-coordinate residual generation for surface-defect augmentation."""

    def __init__(self, config: ContextSGDAConfig | None = None) -> None:
        self.config = config or ContextSGDAConfig()

    def generate(
        self,
        samples: Sequence[DefectSample],
        output_dir: Path,
        *,
        reference_samples: Optional[Sequence[DefectSample]] = None,
    ) -> list[Path]:
        cfg = self.config
        output_dir = Path(output_dir)
        image_dir = output_dir / "images"
        artifacts_dir = output_dir / "artifacts" / "context_sgda"
        candidate_dir = artifacts_dir / "candidates"
        mask_dir = artifacts_dir / "masks"
        edit_mask_dir = artifacts_dir / "edit_masks"
        clean_dir = artifacts_dir / "clean_canvas"
        flow_dir = artifacts_dir / "residual_flow"
        erase_dir = artifacts_dir / "erase_masks"
        field_dir = artifacts_dir / "residual_fields"
        for path in (image_dir, candidate_dir, mask_dir, edit_mask_dir, clean_dir, flow_dir, erase_dir, field_dir):
            path.mkdir(parents=True, exist_ok=True)

        selected_samples = list(samples)
        references = list(reference_samples or selected_samples)
        prior = MorphologyPrior.from_samples(references)
        prior.write_manifest(artifacts_dir / "morphology_prior.json")
        prototype_bank = DRFTResidualPrototypeBank.from_samples(references, target_size=cfg.target_size)
        prototype_bank.write_manifest(artifacts_dir / "residual_prototype_bank.json")

        sampler = AnalyticResidualFlowSampler(steps=cfg.steps)
        rows: list[dict[str, object]] = []
        outputs: list[Path] = []

        for sample_index, sample in enumerate(track(selected_samples, description="Context-SGDA residual generation")):
            target_key = _sample_target_key(sample)
            original = Image.open(sample.image_path).convert("RGB")
            original_size = original.size
            original_resized = original.resize(cfg.target_size, Image.Resampling.LANCZOS)
            scaled_bbox = scale_bbox(sample.bbox, original_size, cfg.target_size)
            protect_mask = _build_instance_protect_mask(sample, original_size, cfg.target_size)
            protected_bboxes = _instance_protect_bboxes(sample, original_size, cfg.target_size)
            best: tuple[float, Image.Image, Image.Image, Image.Image, Image.Image, dict[str, object]] | None = None
            sample_rows: list[dict[str, object]] = []

            for candidate_index in range(max(1, int(cfg.candidates_per_sample))):
                candidate_seed = int(cfg.seed + sample_index * 1777 + candidate_index * 131)
                mask, plan = build_morphology_calibrated_mask(
                    prior,
                    sample.cls_name,
                    sample.bbox,
                    target_key=target_key,
                    target_size=cfg.target_size,
                    target_source_size=original_size,
                    candidate_index=candidate_index,
                    seed=candidate_seed,
                    feather_radius=cfg.feather_radius,
                )
                mask = _subtract_protect_mask(mask, protect_mask)
                clean_canvas, erase_mask, canvas_meta = build_adaptive_counterfactual_canvas(
                    original_resized,
                    scaled_bbox,
                    sample.cls_name,
                    seed=candidate_seed,
                    candidates=5,
                )
                erase_mask = _subtract_protect_mask(erase_mask, protect_mask)
                background_canvas = create_composite_preserving_background(original_resized, clean_canvas, erase_mask)
                prototype = prototype_bank.sample(sample.cls_name, seed=candidate_seed, avoid_key=target_key)
                field = build_class_aware_defect_residual_field(
                    background_canvas,
                    mask,
                    sample.cls_name,
                    orientation_deg=plan.orientation_deg,
                    seed=candidate_seed,
                    prototype=prototype,
                )
                contract = build_context_preserved_drft_contract(
                    mask,
                    field,
                    scaled_bbox,
                    cfg.target_size,
                    protect_mask=protect_mask,
                    context_dilation=cfg.context_dilation,
                    shell_weight=cfg.context_shell_weight,
                    core_edit_weight=cfg.context_core_weight,
                    core_threshold=cfg.context_core_threshold,
                    variant_name="context-sgda-residual",
                )
                flow_image, frames = sampler.sample(background_canvas, field, seed=candidate_seed, return_frames=False)
                final_resized = create_composite_preserving_background(original_resized, flow_image, contract.edit_mask)
                final_native = create_composite_preserving_background(original, final_resized, contract.edit_mask)

                candidate_name = f"{target_key}_c{candidate_index:02d}"
                candidate_path = candidate_dir / f"{candidate_name}.png"
                mask_path = mask_dir / f"{candidate_name}_mask.png"
                edit_mask_path = edit_mask_dir / f"{candidate_name}_edit.png"
                clean_path = clean_dir / f"{candidate_name}_clean.png"
                flow_path = flow_dir / f"{candidate_name}_flow.png"
                erase_path = erase_dir / f"{candidate_name}_erase.png"
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                final_native.save(candidate_path)
                mask.save(mask_path)
                contract.edit_mask.save(edit_mask_path)
                background_canvas.save(clean_path)
                flow_image.save(flow_path)
                erase_mask.save(erase_path)
                field_paths = save_drft_artifacts(field_dir, candidate_name, field, frames)

                quality = score_drft_candidate(background_canvas, final_resized, field)
                context_quality = score_drft_context_contract(original_resized, final_resized, contract)
                selection_score, selection_terms = _selection_score(quality, context_quality, field, contract.edit_mask)
                row: dict[str, object] = {
                    "target_key": target_key,
                    "source_key": sample.source_stem,
                    "object_index": int(getattr(sample, "object_index", 0)),
                    "object_count": len(getattr(sample, "objects", ()) or ()) or 1,
                    "protected_bboxes": [list(box) for box in protected_bboxes],
                    "class": sample.cls_name,
                    "candidate_index": int(candidate_index),
                    "candidate_path": str(candidate_path.resolve()),
                    "selected": False,
                    "method": "context_sgda_residual",
                    "paper_reference": "Context-SGDA: context-coordinate residual defect generation",
                    "spatial_code": {
                        "source_size": [int(original_size[0]), int(original_size[1])],
                        "target_size": [int(cfg.target_size[0]), int(cfg.target_size[1])],
                        "source_bbox": [int(v) for v in sample.bbox],
                        "scaled_bbox": [int(v) for v in scaled_bbox],
                        "context_expanded_bbox": [int(v) for v in contract.expanded_bbox],
                    },
                    "morphology_code": asdict(plan),
                    "texture_code": {
                        "family": field.family,
                        "orientation_deg": float(field.orientation_deg),
                        "prototype_key": prototype.key if prototype is not None else None,
                        "residual_min": float(np.min(field.signed_residual)),
                        "residual_max": float(np.max(field.signed_residual)),
                        "residual_mean_abs": _mean_abs_residual(field),
                    },
                    "residual_generation": {
                        "form": "I_syn = I_real + A(context, box) * delta_defect",
                        "steps": int(cfg.steps),
                        "sampler": "analytic_residual_flow",
                    },
                    "canvas": canvas_meta,
                    "context_contract": {"enabled": True, **contract.stats},
                    "selection_score": float(selection_score),
                    "selection_terms": selection_terms,
                    "quality": asdict(quality),
                    "context_quality": asdict(context_quality),
                    "paths": {
                        "mask": str(mask_path.resolve()),
                        "edit_mask": str(edit_mask_path.resolve()),
                        "clean_canvas": str(clean_path.resolve()),
                        "residual_flow": str(flow_path.resolve()),
                        "erase_mask": str(erase_path.resolve()),
                        "residual_field": field_paths,
                    },
                    "label_contract": "source-label-inherited",
                }
                sample_rows.append(row)
                if best is None or selection_score > best[0]:
                    best = (
                        selection_score,
                        final_native,
                        mask,
                        contract.edit_mask,
                        erase_mask,
                        row,
                    )

            if best is None:
                continue
            _, selected_image, selected_mask, selected_edit_mask, selected_erase_mask, selected_row = best
            selected_row["selected"] = True
            out_path = image_dir / f"{target_key}.png"
            selected_image.save(out_path)
            selected_mask.save(mask_dir / f"{target_key}_selected_mask.png")
            selected_edit_mask.save(edit_mask_dir / f"{target_key}_selected_edit.png")
            selected_erase_mask.save(erase_dir / f"{target_key}_selected_erase.png")
            selected_row["output_path"] = str(out_path.resolve())
            rows.extend(sample_rows)
            outputs.append(out_path)

        score_path = artifacts_dir / "candidate_scores.jsonl"
        append_jsonl(score_path, rows)
        manifest = {
            "method": "context_sgda_residual",
            "generated_images": len(outputs),
            "image_dir": str(image_dir.resolve()),
            "score_path": str(score_path.resolve()),
            "config": asdict(cfg),
            "innovation": {
                "context_coordinate_generation": "spatial, morphology and texture codes condition the residual field",
                "residual_context_preservation": "synthetic defects are added as local residual perturbations under a context contract",
            },
        }
        (output_dir / "context_sgda_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return outputs
