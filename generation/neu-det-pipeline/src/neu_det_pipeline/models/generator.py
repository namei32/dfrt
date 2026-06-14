from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from PIL import Image, ImageDraw, ImageFilter
from rich.progress import track

from ..config import GenerationConfig
from ..data.loader import DefectSample
from ..guidance.morphology import (
    MorphologyPrior,
    build_morphology_calibrated_mask,
    scale_bbox,
)
from ..guidance.drft import (
    DRFTResidualPrototypeBank,
    append_jsonl as append_drft_jsonl,
    build_adaptive_counterfactual_canvas,
    build_class_aware_defect_residual_field,
    build_context_preserved_drft_contract,
    build_residual_seeded_canvas,
    save_drft_artifacts,
    score_residual_bridge_alignment,
    score_drft_context_contract,
    score_drft_candidate,
    summarize_counterfactual_residual_bridge,
)
from ..guidance.mask import create_composite_preserving_background
from .drft_lora import (
    DRFTAttentionContext,
    DRFTAttnProcessor2_0,
    is_drft_lora_path,
    load_drft_lora_into_unet,
    read_drft_lora_metadata,
    residual_field_to_tensor,
)

logger = logging.getLogger(__name__)

SCHEDULER_REGISTRY = {
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "DDIMScheduler": DDIMScheduler,
    "PNDMScheduler": PNDMScheduler,
}


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


def _residual_concentration(field: Any, edit_mask: Image.Image) -> float:
    residual = np.abs(np.asarray(field.signed_residual, dtype=np.float32))
    mask = np.asarray(edit_mask.resize((residual.shape[1], residual.shape[0]), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255.0
    total = float(np.sum(residual) + 1e-6)
    inside = float(np.sum(residual * np.clip(mask, 0.0, 1.0)))
    return max(0.0, min(1.0, inside / total))


def _resize_l_mask(mask: Image.Image, size: tuple[int, int], *, soft: bool = True) -> Image.Image:
    resample = Image.Resampling.LANCZOS if soft else Image.Resampling.NEAREST
    return mask.convert("L").resize(size, resample)


def _defect_first_selection(
    quality: Any,
    context_quality: Any | None,
    field: Any,
    edit_mask: Image.Image,
    *,
    min_context_score: float,
) -> tuple[float, dict[str, object]]:
    inside_outside_ratio = float(quality.inside_delta / max(quality.outside_delta, 1e-4))
    ratio_score = 1.0 - float(np.exp(-inside_outside_ratio / 4.0))
    concentration = _residual_concentration(field, edit_mask)
    defect_score = (
        0.46 * float(quality.defect_strength_score)
        + 0.24 * float(quality.boundary_score)
        + 0.18 * ratio_score
        + 0.12 * concentration
    )
    context_total = float(context_quality.total) if context_quality is not None else 1.0
    context_ok = context_total >= float(min_context_score)
    penalty = max(0.0, float(min_context_score) - context_total) * 2.5
    score = float(defect_score - penalty)
    meta = {
        "mode": "context-gated-defect-first",
        "context_ok": bool(context_ok),
        "min_context_score": float(min_context_score),
        "context_total": float(context_total),
        "defect_score": float(defect_score),
        "inside_outside_ratio": float(inside_outside_ratio),
        "ratio_score": float(ratio_score),
        "residual_concentration": float(concentration),
        "context_penalty": float(penalty),
    }
    return score, meta


class DRFTGenerator:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg

    @staticmethod
    def _safe_empty_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _require_discrete_cuda(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for generation in this pipeline.")
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        logger.info("Using CUDA device %d: %s (%.1f GB)", idx, props.name, props.total_memory / (1024 ** 3))
        return torch.device(f"cuda:{idx}")

    def _auth_token(self) -> Optional[str]:
        return self.cfg.hf_token or os.getenv("HF_TOKEN")

    def _configure_scheduler(self, pipe: Any) -> None:
        scheduler_cls = SCHEDULER_REGISTRY.get(self.cfg.scheduler)
        if scheduler_cls is None:
            raise ValueError(f"Unknown scheduler: {self.cfg.scheduler}")
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)

    def _load_textual_inversion_embeddings(self, pipe: Any, embeddings_dir: Path) -> None:
        tokenizer = getattr(pipe, "tokenizer", None)
        text_encoder = getattr(pipe, "text_encoder", None)
        if tokenizer is None or text_encoder is None:
            return
        for emb_file in Path(embeddings_dir).glob("*_embedding.pt"):
            cls_name = emb_file.stem.replace("_embedding", "")
            token = f"<neu_{cls_name}>"
            checkpoint = torch.load(emb_file, map_location=text_encoder.device)
            embedding = checkpoint["embedding"] if isinstance(checkpoint, dict) and "embedding" in checkpoint else checkpoint
            if not hasattr(embedding, "shape"):
                continue
            added = tokenizer.add_tokens(token)
            if added:
                text_encoder.resize_token_embeddings(len(tokenizer))
            token_id = tokenizer.convert_tokens_to_ids(token)
            text_encoder.get_input_embeddings().weight.data[token_id] = embedding.to(text_encoder.device)

    def _release_pipeline(self, pipe: Any) -> None:
        try:
            pipe.to("cpu")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping CPU transfer: %s", exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _sample_generator(self, pipe: Any, seed: int) -> torch.Generator:
        execution_device = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", torch.device("cpu"))
        if isinstance(execution_device, str):
            execution_device = torch.device(execution_device)
        if isinstance(execution_device, torch.device) and execution_device.type == "meta":
            execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.Generator(device=execution_device).manual_seed(int(seed))

    def _build_drft_lora_pipeline(self, lora_path: Path) -> tuple[Any, DRFTAttentionContext, Dict[str, Any]]:
        if StableDiffusionInpaintPipeline is None:
            raise RuntimeError("StableDiffusionInpaintPipeline is unavailable.")
        if not is_drft_lora_path(lora_path):
            raise ValueError(f"{lora_path} is not a DRFT-LoRA adapter.")

        self._require_discrete_cuda()
        torch.cuda.empty_cache()
        metadata = read_drft_lora_metadata(lora_path)
        model_id = str(metadata.get("model_id") or self.cfg.base_model or "runwayml/stable-diffusion-inpainting")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            use_auth_token=self._auth_token(),
        )
        self._configure_scheduler(pipe)
        embeddings_dir = Path(lora_path).parent.parent / "textual_inversion"
        if embeddings_dir.exists():
            self._load_textual_inversion_embeddings(pipe, embeddings_dir)
        try:
            pipe.disable_xformers_memory_efficient_attention()
        except Exception:  # noqa: BLE001
            pass
        context, metadata = load_drft_lora_into_unet(pipe.unet, lora_path)
        dtype = getattr(pipe.unet, "dtype", torch.float16)
        for processor in pipe.unet.attn_processors.values():
            if isinstance(processor, torch.nn.Module):
                processor.to(dtype=dtype)
                processor.eval()
        if torch.cuda.is_available():
            # DRFT installs custom attention processors after the pipeline is
            # created. Keeping the full pipeline on CUDA preserves those module
            # weights during inference; CPU offload can bypass their effect.
            pipe.to("cuda")
        drft_processor_count = sum(
            isinstance(processor, DRFTAttnProcessor2_0)
            for processor in pipe.unet.attn_processors.values()
        )
        if drft_processor_count <= 0:
            raise RuntimeError("DRFT-LoRA attention processors were not installed.")
        pipe.unet.eval()
        return pipe, context, metadata

    def generate_drft_lora_inpaint(
        self,
        lora_path: Path,
        prompt_items: List[tuple[str, str, str]],
        conditioning: Dict[str, Dict[str, Path]],
        output_dir: Path,
        init_images: Dict[str, Path],
        bbox_map: Dict[str, tuple[int, int, int, int]],
        samples: Sequence[DefectSample],
        *,
        reference_samples: Optional[Sequence[DefectSample]] = None,
        candidates_per_sample: int = 2,
        quality_threshold: float = 0.0,
        variant: str = "v2",
        context_dilation: float = 1.65,
        context_shell_weight: float = 0.55,
        context_core_edit_weight: float = 1.0,
        context_core_threshold: float = 0.10,
        residual_seed_gain: float = 0.0,
        context_min_score: float = 0.985,
        defect_first_selection: bool = True,
    ) -> List[Path]:
        """Full DRFT-LoRA: residual-field/timestep-gated LoRA inside SD inpainting U-Net."""

        _ = conditioning
        output_dir.mkdir(parents=True, exist_ok=True)
        pipe, context, metadata = self._build_drft_lora_pipeline(lora_path)

        variant_lc = variant.lower().replace("_", "-")
        use_no_context_boost = variant_lc in {
            "v2-nocontext",
            "v2-no-context",
            "drft-v2-nocontext",
            "drft-v2-no-context",
        }
        use_context_contract = variant_lc in {
            "v2-context",
            "drft-v2-context",
            "drft-v2-cprt",
            "cprt",
            "context-preserved-drft",
        }
        use_plain_v2 = variant_lc in {"v2", "drft-v2", "drft-v2-lora"}
        if not (use_context_contract or use_no_context_boost or use_plain_v2):
            raise ValueError(f"Unsupported retained generation variant: {variant}")
        artifact_name = (
            "drft_v2_context_lora"
            if use_context_contract
            else ("drft_v2_nocontext_lora" if use_no_context_boost else "drft_v2_lora")
        )
        artifacts_dir = output_dir.parent / "artifacts" / artifact_name
        mask_dir = artifacts_dir / "masks"
        edit_mask_dir = artifacts_dir / "edit_masks"
        context_mask_dir = artifacts_dir / "context_masks"
        protect_mask_dir = artifacts_dir / "protect_masks"
        seed_dir = artifacts_dir / "seed_canvas"
        erase_dir = artifacts_dir / "erase_masks"
        clean_dir = artifacts_dir / "clean_canvas"
        candidate_dir = artifacts_dir / "candidates"
        field_dir = artifacts_dir / "residual_fields"
        for path in (mask_dir, edit_mask_dir, context_mask_dir, protect_mask_dir, seed_dir, erase_dir, clean_dir, candidate_dir, field_dir):
            path.mkdir(parents=True, exist_ok=True)

        class_to_id = metadata.get("class_to_id") or {}
        if not class_to_id:
            class_to_id = {name: idx for idx, name in enumerate(metadata.get("class_names") or [])}
        target_size = (512, 512)
        prior = MorphologyPrior.from_samples(reference_samples or samples)
        prior.write_manifest(artifacts_dir / "morphology_prior.json")
        prototype_bank = DRFTResidualPrototypeBank.from_samples(reference_samples or samples, target_size=target_size)
        prototype_bank.write_manifest(artifacts_dir / "residual_prototype_bank.json")
        score_path = artifacts_dir / "candidate_scores.jsonl"
        sample_by_stem = {_sample_target_key(sample): sample for sample in samples}
        candidates_per_sample = max(1, int(candidates_per_sample))
        results: List[Path] = []

        try:
            for idx, item in enumerate(track(prompt_items, description="DRFT-LoRA inpaint")):
                stem, base_prompt = item[0], item[1]
                cls_name = item[2]
                if stem not in init_images or stem not in bbox_map:
                    continue
                target_sample = sample_by_stem.get(stem)
                if target_sample is None:
                    continue

                orig_img = Image.open(init_images[stem]).convert("RGB")
                orig_size = orig_img.size
                orig_resized = orig_img.resize(target_size, Image.Resampling.LANCZOS)
                scaled_bbox = scale_bbox(bbox_map[stem], orig_size, target_size)
                protect_mask = _build_instance_protect_mask(target_sample, orig_size, target_size)
                protected_bboxes = _instance_protect_bboxes(target_sample, orig_size, target_size)
                best: tuple[
                    float,
                    Image.Image,
                    Image.Image,
                    Image.Image,
                    Image.Image,
                    Image.Image,
                    dict[str, object],
                ] | None = None
                sample_score_rows: list[dict[str, object]] = []

                for candidate_index in range(candidates_per_sample):
                    candidate_seed = int(self.cfg.seed + idx * 1777 + candidate_index * 131)
                    mask, plan = build_morphology_calibrated_mask(
                        prior,
                        cls_name,
                        bbox_map[stem],
                        target_key=stem,
                        target_size=target_size,
                        target_source_size=orig_size,
                        candidate_index=candidate_index,
                        seed=candidate_seed,
                        feather_radius=1.4,
                    )
                    mask = _subtract_protect_mask(mask, protect_mask)
                    clean_canvas, erase_mask, canvas_meta = build_adaptive_counterfactual_canvas(
                        orig_resized,
                        scaled_bbox,
                        cls_name,
                        seed=candidate_seed,
                        candidates=5,
                    )
                    erase_mask = _subtract_protect_mask(erase_mask, protect_mask)
                    background_canvas = create_composite_preserving_background(orig_resized, clean_canvas, erase_mask)
                    prototype = prototype_bank.sample(cls_name, seed=candidate_seed, avoid_key=stem)
                    field = build_class_aware_defect_residual_field(
                        background_canvas,
                        mask,
                        cls_name,
                        orientation_deg=plan.orientation_deg,
                        seed=candidate_seed,
                        prototype=prototype,
                    )
                    residual_bridge = summarize_counterfactual_residual_bridge(
                        field,
                        target_bbox=scaled_bbox,
                        target_size=target_size,
                    )
                    context_contract = None
                    generation_mask = mask
                    inpaint_canvas = background_canvas
                    seed_meta: dict[str, object] = {"enabled": False}
                    effective_residual_seed_gain = float(residual_seed_gain)
                    if use_context_contract:
                        context_contract = build_context_preserved_drft_contract(
                            mask,
                            field,
                            scaled_bbox,
                            target_size,
                            protect_mask=protect_mask,
                            context_dilation=context_dilation,
                            shell_weight=context_shell_weight,
                            core_edit_weight=context_core_edit_weight,
                            core_threshold=context_core_threshold,
                            variant_name="drft-v2-context",
                        )
                        generation_mask = context_contract.edit_mask
                        inpaint_canvas, seed_meta = build_residual_seeded_canvas(
                            background_canvas,
                            field,
                            residual_seed_gain=effective_residual_seed_gain,
                        )
                    elif effective_residual_seed_gain > 0.0:
                        inpaint_canvas, seed_meta = build_residual_seeded_canvas(
                            background_canvas,
                            field,
                            residual_seed_gain=effective_residual_seed_gain,
                        )
                    class_id_fallback = cls_name not in class_to_id
                    class_id = int(class_to_id.get(cls_name, 0))
                    context.set_condition(
                        residual_field_to_tensor(field).unsqueeze(0),
                        torch.tensor([class_id], dtype=torch.long),
                    )
                    generator = self._sample_generator(pipe, candidate_seed)
                    denoising_strength = self.cfg.denoising_strength
                    context.begin_trace(max_events=96)
                    try:
                        with torch.inference_mode():
                            output = pipe(
                                prompt=base_prompt,
                                image=inpaint_canvas,
                                mask_image=generation_mask,
                                num_inference_steps=self.cfg.num_inference_steps,
                                guidance_scale=self.cfg.guidance_scale,
                                strength=denoising_strength,
                                generator=generator,
                            )
                    finally:
                        adaptation_trace = context.end_trace()
                    if adaptation_trace.get("enabled") and int(adaptation_trace.get("event_count") or 0) == 0:
                        logger.warning(
                            "DRFT residual-bridge adaptation trace recorded no events for %s candidate %d",
                            stem,
                            candidate_index,
                        )
                    generated = output.images[0].resize(target_size, Image.Resampling.LANCZOS)
                    final_image = create_composite_preserving_background(orig_resized, generated, generation_mask)
                    native_final_image = create_composite_preserving_background(orig_img, generated, generation_mask)
                    quality = score_drft_candidate(background_canvas, final_image, field)
                    residual_bridge_alignment = score_residual_bridge_alignment(background_canvas, final_image, field)
                    context_quality = (
                        score_drft_context_contract(orig_resized, final_image, context_contract)
                        if context_contract is not None
                        else None
                    )
                    selection_score = 0.86 * float(quality.total) + 0.14 * float(residual_bridge_alignment.total)
                    selection_meta: dict[str, object] = {
                        "mode": "quality-bridge",
                        "quality_total": float(quality.total),
                        "residual_bridge_alignment_total": float(residual_bridge_alignment.total),
                    }
                    if (
                        defect_first_selection
                        and (context_quality is not None or use_no_context_boost or residual_seed_gain > 0.0)
                    ):
                        defect_selection_score, selection_meta = _defect_first_selection(
                            quality,
                            context_quality,
                            field,
                            generation_mask,
                            min_context_score=context_min_score if context_quality is not None else 0.0,
                        )
                        selection_score = (
                            0.86 * float(defect_selection_score)
                            + 0.14 * float(residual_bridge_alignment.total)
                        )
                        selection_meta = {
                            **selection_meta,
                            "base_selection_score": float(defect_selection_score),
                            "residual_bridge_alignment_total": float(residual_bridge_alignment.total),
                        }
                    elif context_quality is not None:
                        selection_score = (
                            0.62 * float(quality.total)
                            + 0.24 * float(context_quality.total)
                            + 0.14 * float(residual_bridge_alignment.total)
                        )
                        selection_meta = {
                            "mode": "weighted-quality-context-bridge",
                            "quality_total": float(quality.total),
                            "context_total": float(context_quality.total),
                            "residual_bridge_alignment_total": float(residual_bridge_alignment.total),
                        }
                    candidate_name = f"{stem}_c{candidate_index:02d}"
                    candidate_path = candidate_dir / f"{candidate_name}.png"
                    mask_path = mask_dir / f"{candidate_name}_mask.png"
                    edit_mask_path = edit_mask_dir / f"{candidate_name}_edit.png"
                    context_mask_path = context_mask_dir / f"{candidate_name}_context.png"
                    protect_mask_path = protect_mask_dir / f"{candidate_name}_protect.png"
                    seed_path = seed_dir / f"{candidate_name}_seed.png"
                    erase_path = erase_dir / f"{candidate_name}_erase.png"
                    clean_path = clean_dir / f"{candidate_name}_clean.png"
                    final_image.save(candidate_path)
                    mask.save(mask_path)
                    generation_mask.save(edit_mask_path)
                    inpaint_canvas.save(seed_path)
                    if context_contract is not None:
                        context_contract.context_mask.save(context_mask_path)
                        context_contract.protect_mask.save(protect_mask_path)
                    erase_mask.save(erase_path)
                    background_canvas.save(clean_path)
                    field_paths = save_drft_artifacts(field_dir, candidate_name, field, [])

                    row = {
                        "target_key": stem,
                        "source_key": target_sample.image_path.stem,
                        "object_index": int(getattr(target_sample, "object_index", 0)),
                        "object_count": len(getattr(target_sample, "objects", ()) or ()) or 1,
                        "protected_bboxes": [list(box) for box in protected_bboxes],
                        "class": cls_name,
                        "class_id": class_id,
                        "class_id_fallback": bool(class_id_fallback),
                        "prompt": base_prompt,
                        "candidate_index": candidate_index,
                        "candidate_path": str(candidate_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "edit_mask_path": str(edit_mask_path.resolve()),
                        "seed_canvas_path": str(seed_path.resolve()),
                        "erase_mask_path": str(erase_path.resolve()),
                        "clean_canvas_path": str(clean_path.resolve()),
                        "prototype_key": prototype.key if prototype is not None else None,
                        "adapter_path": str(Path(lora_path).resolve()),
                        "drft_variant": (
                            "drft-v2-context"
                            if use_context_contract
                            else ("drft-v2-nocontext" if use_no_context_boost else "drft-v2")
                        ),
                        "denoising_strength": float(denoising_strength),
                        "canvas": canvas_meta,
                        "residual_seed": seed_meta,
                        "context_contract": (
                            {
                                "enabled": True,
                                "context_mask_path": str(context_mask_path.resolve()),
                                "protect_mask_path": str(protect_mask_path.resolve()),
                                **context_contract.stats,
                            }
                            if context_contract is not None
                            else {"enabled": False}
                        ),
                        "selection_score": selection_score,
                        "selection_policy": selection_meta,
                        "selected": False,
                        **asdict(plan),
                        "residual_field": field_paths,
                        "residual_bridge": asdict(residual_bridge),
                        "residual_bridge_alignment": asdict(residual_bridge_alignment),
                        "local_diffusion_adaptation": adaptation_trace,
                        "quality": asdict(quality),
                        "context_quality": asdict(context_quality) if context_quality is not None else None,
                        "native_resolution": {
                            "enabled": True,
                            "source_size": [int(orig_size[0]), int(orig_size[1])],
                            "working_size": [int(target_size[0]), int(target_size[1])],
                            "output_size": [int(native_final_image.size[0]), int(native_final_image.size[1])],
                            "composition": "resize_generated_candidate_under_edit_mask_only",
                        },
                    }
                    sample_score_rows.append(row)
                    if best is None or selection_score > best[0]:
                        best = (selection_score, native_final_image, mask, generation_mask, erase_mask, background_canvas, row)

                if best is None:
                    continue
                selected_score, selected_image, selected_mask, selected_edit_mask, selected_erase, selected_clean, selected_row = best
                if selected_score < float(quality_threshold):
                    selected_row["below_quality_threshold"] = True
                selected_row["selected"] = True
                out_path = output_dir / f"{stem}.png"
                selected_image.save(out_path)
                selected_mask.save(mask_dir / f"{stem}_selected_mask_512.png")
                selected_edit_mask.save(edit_mask_dir / f"{stem}_selected_edit_512.png")
                selected_erase.save(erase_dir / f"{stem}_selected_erase_512.png")
                _resize_l_mask(selected_mask, selected_image.size).save(mask_dir / f"{stem}_selected_mask.png")
                _resize_l_mask(selected_edit_mask, selected_image.size).save(edit_mask_dir / f"{stem}_selected_edit.png")
                _resize_l_mask(selected_erase, selected_image.size).save(erase_dir / f"{stem}_selected_erase.png")
                selected_clean.save(clean_dir / f"{stem}_selected_clean.png")
                append_drft_jsonl(score_path, sample_score_rows)
                results.append(out_path)
        finally:
            self._release_pipeline(pipe)

        return results
