from __future__ import annotations

import json
import logging
import random
import shutil
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from .config import ConfigBundle, load_config_bundle
from .data import (
    DefectSample,
    collect_dataset_images,
    collect_dataset_instances,
    compute_class_counts,
    create_mixed_dataset,
    export_metadata,
    split_dataset,
)
from .models import (
    BackgroundProjectorConfig,
    BackgroundProjectorTrainer,
    DRFTGenerator,
    DRFTLoRAHyperParams,
    DRFTLoRATrainer,
)
from .models.drft_lora import is_drft_lora_path, read_drft_lora_metadata
from .prompts import CaptionGenerator, generate_captions_with_blip2, load_captions_from_file

app = typer.Typer(
    add_completion=False,
    help="Retained DRFT-v2 pipeline: DRFT-LoRA training, native generation, mixed data, YOLO.",
)
console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workspace_root() -> Path:
    return _project_root().parents[1]


def _default_split_manifest() -> Path:
    return _project_root() / "data" / "processed" / "split_manifest.json"


def _ensure_bundle(ctx: typer.Context) -> ConfigBundle:
    if isinstance(ctx.obj, ConfigBundle):
        return ctx.obj
    bundle = load_config_bundle()
    ctx.obj = bundle
    return bundle


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _sample_target_key(sample: DefectSample) -> str:
    return str(getattr(sample, "target_key", sample.image_path.stem))


def _load_split_stems(manifest_path: Optional[Path], split: str) -> set[str] | None:
    if not split or split.lower() == "all":
        return None
    if manifest_path is None or not Path(manifest_path).exists():
        return None
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    values = payload.get(split)
    if values is None:
        raise typer.BadParameter(f"Split '{split}' was not found in {manifest_path}.")
    return {str(item) for item in values}


def _filter_samples_by_stems(samples: list[DefectSample], stems: set[str] | None) -> list[DefectSample]:
    if stems is None:
        return list(samples)
    kept: list[DefectSample] = []
    for sample in samples:
        keys = {
            sample.image_path.stem,
            sample.source_stem,
            _sample_target_key(sample),
        }
        if keys & stems:
            kept.append(sample)
    return kept


def _filter_samples_for_split(
    samples: list[DefectSample],
    split: str,
    split_manifest: Optional[Path],
) -> list[DefectSample]:
    stems = _load_split_stems(split_manifest, split)
    return _filter_samples_by_stems(samples, stems)


def _balanced_take(samples: list[DefectSample], limit: int) -> list[DefectSample]:
    groups: dict[str, list[DefectSample]] = {}
    for sample in samples:
        groups.setdefault(sample.cls_name, []).append(sample)
    selected: list[DefectSample] = []
    class_names = sorted(groups)
    cursor = 0
    while len(selected) < limit and any(groups.values()):
        cls_name = class_names[cursor % len(class_names)]
        if groups[cls_name]:
            selected.append(groups[cls_name].pop(0))
        cursor += 1
    return selected


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _find_first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _infer_orig_images_dir(dataset_root: Path, explicit: Optional[Path]) -> Path | None:
    if explicit is not None:
        return explicit
    return _find_first_existing(
        [
            dataset_root / "IMAGES",
            dataset_root / "images",
            dataset_root,
        ]
    )


def _infer_orig_labels_dir(dataset_root: Path, explicit: Optional[Path]) -> Path | None:
    if explicit is not None:
        return explicit
    project = _project_root()
    workspace = _workspace_root()
    return _find_first_existing(
        [
            dataset_root / "labels",
            dataset_root.parent / "labels",
            project / "data" / "processed" / "yolo" / "labels",
            workspace / "data" / "NEU" / "split" / "neu_det_split" / "labels",
        ]
    )


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    result: dict[str, Any] = {}
    for key, attr in {
        "mAP50": "map50",
        "mAP50-95": "map",
        "precision": "mp",
        "recall": "mr",
    }.items():
        value = getattr(box, attr, None) if box is not None else None
        try:
            result[key] = float(value) if value is not None else None
        except (TypeError, ValueError):
            result[key] = None
    precision = result.get("precision")
    recall = result.get("recall")
    if precision is not None and recall is not None and precision + recall > 0:
        result["f1"] = 2.0 * precision * recall / (precision + recall)
    else:
        result["f1"] = None
    return result


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Deprecated; retained for CLI compatibility. Dataclass defaults are used.",
    ),
) -> None:
    ctx.obj = load_config_bundle(config)


@app.command("prepare")
def prepare_dataset(
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("data/processed"), "--output-dir", file_okay=False),
    test_size: float = typer.Option(0.1, "--test-size"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Create a lightweight image-level split manifest for DRFT-v2 runs."""

    samples = collect_dataset_images(dataset_root)
    if not samples:
        raise typer.BadParameter(f"No Pascal VOC image samples found under {dataset_root}.")

    try:
        splits = split_dataset(samples, test_size=test_size, seed=seed)
        train_samples = splits.train
        val_samples = splits.val
    except ValueError:
        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        cut = max(1, int(round(len(shuffled) * (1.0 - test_size))))
        train_samples = shuffled[:cut]
        val_samples = shuffled[cut:]

    manifest = {
        "train": sorted(sample.source_stem for sample in train_samples),
        "val": sorted(sample.source_stem for sample in val_samples),
        "test": [],
        "seed": int(seed),
        "dataset_root": str(dataset_root.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "split_manifest.json", manifest)
    export_metadata(samples, output_dir / "metadata.json")
    console.print(f"[green]Prepared {len(samples)} image-level samples at {output_dir}[/green]")
    console.print(f"[cyan]Class counts: {dict(sorted(compute_class_counts(samples).items()))}[/cyan]")


@app.command("caption")
def caption_dataset(
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_file: Path = typer.Option(Path("outputs/captions.json"), "--output-file"),
    model_name: str = typer.Option("openai/clip-vit-large-patch14", "--model-name"),
    use_blip2: bool = typer.Option(False, "--blip2/--clip"),
    blip2_model: str = typer.Option("Salesforce/blip2-opt-2.7b", "--blip2-model"),
    use_clip_selection: bool = typer.Option(False, "--clip-selection/--no-clip-selection"),
    use_paper_keywords: bool = typer.Option(True, "--paper-keywords/--no-paper-keywords"),
    lora_weight: float = typer.Option(1.0, "--lora-weight"),
) -> None:
    """Generate prompt captions for DRFT-v2 generation."""

    samples = collect_dataset_images(dataset_root)
    if not samples:
        raise typer.BadParameter(f"No Pascal VOC image samples found under {dataset_root}.")
    token_map = {cls: f"<neu_{cls}>" for cls in sorted({sample.cls_name for sample in samples})}
    if use_blip2:
        captions = generate_captions_with_blip2(
            samples=samples,
            token_map=token_map,
            output_file=output_file,
            use_paper_keywords=use_paper_keywords,
            lora_weight=lora_weight,
            model_name=blip2_model,
            combine_with_keywords=True,
        )
    else:
        generator = CaptionGenerator(model_name=model_name)
        try:
            captions = generator.generate_with_token(
                samples,
                token_map,
                output_file=output_file,
                use_paper_keywords=use_paper_keywords,
                lora_weight=lora_weight,
                use_clip_selection=use_clip_selection,
            )
        finally:
            generator.cleanup()
    console.print(f"[green]Saved {len(captions)} captions to {output_file}[/green]")


@app.command("train-drft")
def train_drft_lora(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/drft_lora"), "--output-dir", file_okay=False),
    split_manifest: Optional[Path] = typer.Option(None, "--split-manifest"),
    train_split: str = typer.Option("train", "--train-split"),
    steps: int = typer.Option(100, "--steps"),
    batch_size: int = typer.Option(1, "--batch-size"),
    grad_accum: int = typer.Option(4, "--grad-accum"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    resolution: int = typer.Option(512, "--resolution"),
    rank: int = typer.Option(4, "--rank"),
    alpha: int = typer.Option(4, "--alpha"),
    num_experts: int = typer.Option(3, "--num-experts"),
    max_gate: float = typer.Option(1.25, "--max-gate"),
    model_id: str = typer.Option("runwayml/stable-diffusion-inpainting", "--model-id"),
    max_train_samples: Optional[int] = typer.Option(None, "--max-train-samples"),
) -> None:
    """Train DRFT-LoRA on defect residual fields."""

    bundle = _ensure_bundle(ctx)
    cfg = replace(bundle.lora)
    cfg.model_id = model_id
    cfg.steps = max(1, int(steps))
    cfg.batch_size = max(1, int(batch_size))
    cfg.gradient_accumulation_steps = max(1, int(grad_accum))
    cfg.learning_rate = float(learning_rate)
    cfg.resolution = max(64, int(resolution))
    cfg.rank = max(1, int(rank))
    cfg.alpha = max(1, int(alpha))

    samples = collect_dataset_instances(dataset_root)
    train_samples = _filter_samples_for_split(samples, train_split, split_manifest)
    if not train_samples:
        raise typer.BadParameter(f"No samples matched split '{train_split}'.")

    token_map = {cls: f"<neu_{cls}>" for cls in sorted({sample.cls_name for sample in train_samples})}
    hparams = DRFTLoRAHyperParams(
        rank=cfg.rank,
        alpha=cfg.alpha,
        num_experts=max(1, int(num_experts)),
        max_gate=float(max_gate),
    )
    console.print(
        f"[cyan]Training DRFT-LoRA on {len(train_samples)} instance target(s): "
        f"{dict(sorted(Counter(sample.cls_name for sample in train_samples).items()))}[/cyan]"
    )
    trainer = DRFTLoRATrainer(cfg, hparams)
    weights = trainer.train(train_samples, token_map, output_dir, max_train_samples=max_train_samples)
    console.print(f"[green]Saved DRFT-LoRA adapter to {weights}[/green]")


@app.command("train-bg-projector")
def train_background_projector(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs/background_projector"), "--output-dir", file_okay=False),
    split_manifest: Optional[Path] = typer.Option(None, "--split-manifest"),
    train_split: str = typer.Option("train", "--train-split"),
    steps: int = typer.Option(500, "--steps"),
    batch_size: int = typer.Option(4, "--batch-size"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate"),
    resolution: int = typer.Option(512, "--resolution"),
    base_channels: int = typer.Option(32, "--base-channels"),
    max_train_samples: Optional[int] = typer.Option(None, "--max-train-samples"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    """Train the box-aware counterfactual background projector for DRFT-v2."""

    bundle = _ensure_bundle(ctx)
    samples = collect_dataset_instances(dataset_root)
    train_samples = _filter_samples_for_split(samples, train_split, split_manifest)
    if not train_samples:
        raise typer.BadParameter(f"No samples matched split '{train_split}'.")
    cfg = BackgroundProjectorConfig(
        resolution=max(64, int(resolution)),
        base_channels=max(8, int(base_channels)),
        learning_rate=float(learning_rate),
        steps=max(1, int(steps)),
        batch_size=max(1, int(batch_size)),
        seed=int(seed if seed is not None else bundle.generation.seed),
    )
    console.print(
        f"[cyan]Training background projector on {len(train_samples)} instance target(s): "
        f"{dict(sorted(Counter(sample.cls_name for sample in train_samples).items()))}[/cyan]"
    )
    trainer = BackgroundProjectorTrainer(cfg)
    checkpoint = trainer.train(
        train_samples,
        output_dir,
        max_train_samples=max_train_samples,
    )
    console.print(f"[green]Saved background projector to {checkpoint}[/green]")


@app.command("generate")
def generate_drft_v2(
    ctx: typer.Context,
    dataset_root: Path = typer.Argument(..., exists=True, file_okay=False),
    guidance_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    lora_path: Path = typer.Argument(..., exists=True),
    output_dir: Path = typer.Option(Path("outputs/generated"), "--output-dir", file_okay=False),
    caption_file: Optional[Path] = typer.Option(None, "--caption-file"),
    priority_class: Optional[str] = typer.Option(None, "--priority-class"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
    generation_split: str = typer.Option("all", "--generation-split"),
    split_manifest: Optional[Path] = typer.Option(None, "--split-manifest"),
    balanced_max_samples: bool = typer.Option(False, "--balanced-max-samples"),
    model_name: str = typer.Option("openai/clip-vit-large-patch14", "--model-name"),
    mode: str = typer.Option("drft-v2", "--mode", case_sensitive=False),
    make_mixed_dataset: bool = typer.Option(
        True,
        "--make-mixed-dataset/--no-mixed-dataset",
    ),
    orig_images_dir: Optional[Path] = typer.Option(None, "--orig-images-dir"),
    orig_labels_dir: Optional[Path] = typer.Option(None, "--orig-labels-dir"),
    mixed_output_dir: Optional[Path] = typer.Option(None, "--mixed-output-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    drft_candidates: int = typer.Option(2, "--drft-candidates"),
    drft_steps: int = typer.Option(18, "--drft-steps"),
    drft_quality_threshold: float = typer.Option(0.0, "--drft-quality-threshold"),
    drft_context_dilation: float = typer.Option(1.65, "--drft-context-dilation"),
    drft_context_shell_weight: float = typer.Option(0.55, "--drft-context-shell-weight"),
    drft_context_core_weight: float = typer.Option(1.0, "--drft-context-core-weight"),
    drft_context_core_threshold: float = typer.Option(0.10, "--drft-context-core-threshold"),
    drft_residual_seed_gain: float = typer.Option(0.0, "--drft-residual-seed-gain"),
    drft_context_min_score: float = typer.Option(0.985, "--drft-context-min-score"),
    drft_defect_first_selection: bool = typer.Option(
        True,
        "--drft-defect-first-selection/--no-drft-defect-first-selection",
    ),
    bg_projector_path: Optional[Path] = typer.Option(None, "--bg-projector-path", exists=True),
    mixed_max_generated: Optional[int] = typer.Option(None, "--mixed-max-generated"),
    mixed_per_source_limit: int = typer.Option(0, "--mixed-per-source-limit"),
    mixed_per_class_limit: int = typer.Option(0, "--mixed-per-class-limit"),
    mixed_quality_threshold: Optional[float] = typer.Option(None, "--mixed-quality-threshold"),
    mixed_min_canvas_drop: Optional[float] = typer.Option(None, "--mixed-min-canvas-drop"),
    mixed_max_outside_delta: Optional[float] = typer.Option(None, "--mixed-max-outside-delta"),
    mixed_selection_strategy: str = typer.Option("random", "--mixed-selection-strategy"),
) -> None:
    """Generate native-resolution defect images with the retained DRFT-v2 path."""

    _ = guidance_dir
    mode_lc = mode.lower().replace("_", "-")
    if mode_lc != "drft-v2":
        raise typer.BadParameter("Only --mode drft-v2 is retained.")
    if not is_drft_lora_path(lora_path):
        raise typer.BadParameter(f"{lora_path} is not a DRFT-LoRA adapter.")

    bundle = _ensure_bundle(ctx)
    cfg = replace(bundle.generation)
    metadata = read_drft_lora_metadata(lora_path)
    cfg.base_model = str(metadata.get("model_id") or cfg.base_model)
    cfg.num_inference_steps = max(4, int(drft_steps))
    if seed is not None:
        cfg.seed = int(seed)

    split_manifest = split_manifest or _default_split_manifest()
    reference_samples = _filter_samples_for_split(
        collect_dataset_images(dataset_root),
        generation_split,
        split_manifest,
    )
    samples = list(reference_samples)
    if not samples:
        raise typer.BadParameter(f"No image-level samples matched split '{generation_split}'.")
    if priority_class:
        samples = sorted(samples, key=lambda item: (item.cls_name != priority_class, item.source_stem))
    if max_samples is not None:
        limit = max(1, int(max_samples))
        samples = _balanced_take(samples, limit) if balanced_max_samples else samples[:limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_drft-v2_{timestamp}"
    image_dir = run_dir / "images"
    artifacts_dir = run_dir / "artifacts"
    image_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir / "run.log")

    token_map = {cls: f"<neu_{cls}>" for cls in sorted({sample.cls_name for sample in samples})}
    if caption_file is None:
        caption_file = output_dir / "captions.json"
    if caption_file.exists():
        captions = load_captions_from_file(caption_file)
    else:
        captioner = CaptionGenerator(model_name=model_name)
        try:
            captions = captioner.generate_with_token(samples, token_map, output_file=caption_file)
        finally:
            captioner.cleanup()

    caption_snapshot = artifacts_dir / caption_file.name
    if caption_file.exists() and caption_snapshot.resolve() != caption_file.resolve():
        shutil.copy2(caption_file, caption_snapshot)

    prompt_items: list[tuple[str, str, str]] = []
    for sample in samples:
        token = token_map[sample.cls_name]
        caption = (
            captions.get(_sample_target_key(sample))
            or captions.get(sample.source_stem)
            or captions.get(sample.image_path.stem)
            or f"macro shot of {token} steel surface"
        )
        prompt_items.append((_sample_target_key(sample), f"{caption}, {token}", sample.cls_name))

    init_images = {_sample_target_key(sample): sample.image_path for sample in samples}
    bbox_map = {_sample_target_key(sample): sample.bbox for sample in samples}
    generator = DRFTGenerator(cfg)
    console.print(
        f"[cyan]DRFT-v2 generation: {len(samples)} image target(s), "
        f"{max(1, int(drft_candidates))} candidate(s) each, seed={cfg.seed}[/cyan]"
    )
    generated_paths = generator.generate_drft_lora_inpaint(
        lora_path,
        prompt_items,
        {},
        image_dir,
        init_images=init_images,
        bbox_map=bbox_map,
        samples=samples,
        reference_samples=reference_samples,
        candidates_per_sample=max(1, int(drft_candidates)),
        quality_threshold=float(drft_quality_threshold),
        variant="drft-v2",
        context_dilation=float(drft_context_dilation),
        context_shell_weight=float(drft_context_shell_weight),
        context_core_edit_weight=float(drft_context_core_weight),
        context_core_threshold=float(drft_context_core_threshold),
        residual_seed_gain=float(drft_residual_seed_gain),
        context_min_score=float(drft_context_min_score),
        defect_first_selection=bool(drft_defect_first_selection),
        background_projector_path=bg_projector_path,
    )

    score_path = artifacts_dir / "drft_v2_lora" / "candidate_scores.jsonl"
    run_context: dict[str, Any] = {
        "timestamp": timestamp,
        "scheme": "DRFT-v2",
        "dataset_root": str(dataset_root.resolve()),
        "lora_path": str(lora_path.resolve()),
        "base_model": cfg.base_model,
        "seed": cfg.seed,
        "generation_split": generation_split,
        "split_manifest": str(split_manifest.resolve()) if split_manifest.exists() else None,
        "generated_images": len(generated_paths),
        "generated_files": [str(path.resolve()) for path in generated_paths],
        "output_directory": str(image_dir.resolve()),
        "caption_file": str(caption_file.resolve()),
        "artifacts": {
            "captions": str(caption_snapshot.resolve()) if caption_snapshot.exists() else None,
            "drft_candidate_scores": str(score_path.resolve()) if score_path.exists() else None,
        },
        "drft": {
            "candidates_per_sample": max(1, int(drft_candidates)),
            "steps": cfg.num_inference_steps,
            "quality_threshold": float(drft_quality_threshold),
            "context_dilation": float(drft_context_dilation),
            "context_shell_weight": float(drft_context_shell_weight),
            "context_core_weight": float(drft_context_core_weight),
            "context_core_threshold": float(drft_context_core_threshold),
            "residual_seed_gain": float(drft_residual_seed_gain),
            "context_min_score": float(drft_context_min_score),
            "defect_first_selection": bool(drft_defect_first_selection),
            "background_projector_path": str(bg_projector_path.resolve()) if bg_projector_path else None,
        },
    }
    _write_json(run_dir / "run_context.json", run_context)

    if not generated_paths:
        console.print("[yellow]No images were generated.[/yellow]")
        return

    if not make_mixed_dataset:
        console.print("[cyan]Skipping mixed dataset creation (--no-mixed-dataset).[/cyan]")
        console.print(f"[green]Generated images: {image_dir}[/green]")
        return

    manifest_for_mixed = split_manifest if split_manifest.exists() else None
    images_for_mixed = _infer_orig_images_dir(dataset_root, orig_images_dir)
    labels_for_mixed = _infer_orig_labels_dir(dataset_root, orig_labels_dir)
    if manifest_for_mixed is None or images_for_mixed is None or labels_for_mixed is None:
        console.print("[yellow]Skipping mixed dataset creation; manifest/images/labels could not be resolved.[/yellow]")
        console.print(f"[green]Generated images: {image_dir}[/green]")
        return

    mixed_dir = mixed_output_dir or (run_dir / "mixed_dataset")
    create_mixed_dataset(
        orig_images_dir=images_for_mixed,
        orig_labels_dir=labels_for_mixed,
        manifest_path=manifest_for_mixed,
        new_images_dir=image_dir,
        run_output_dir=mixed_dir,
        seed=int(cfg.seed),
        score_path=score_path if score_path.exists() else None,
        max_generated=mixed_max_generated,
        per_source_limit=max(0, int(mixed_per_source_limit)),
        per_class_limit=max(0, int(mixed_per_class_limit)),
        quality_threshold=mixed_quality_threshold,
        min_canvas_drop=mixed_min_canvas_drop,
        max_outside_delta=mixed_max_outside_delta,
        selection_strategy=mixed_selection_strategy,
    )
    console.print(f"[green]Generated images: {image_dir}[/green]")
    console.print(f"[green]Mixed dataset: {mixed_dir}[/green]")


@app.command("train-yolo")
def train_yolo(
    data_yaml: Path = typer.Argument(..., exists=True),
    model: str = typer.Option("yolov8n.yaml", "--model"),
    epochs: int = typer.Option(200, "--epochs"),
    patience: int = typer.Option(50, "--patience"),
    batch: int = typer.Option(16, "--batch"),
    imgsz: int = typer.Option(640, "--imgsz"),
    project: Path = typer.Option(Path("runs/drft_v2_yolo"), "--project"),
    name: Optional[str] = typer.Option(None, "--name"),
    device: str = typer.Option("0", "--device"),
    workers: int = typer.Option(0, "--workers"),
    seed: int = typer.Option(42, "--seed"),
    test_after_train: bool = typer.Option(True, "--test-after-train/--no-test-after-train"),
    exist_ok: bool = typer.Option(True, "--exist-ok/--no-exist-ok"),
) -> None:
    """Train and optionally test a YOLO detector on a DRFT-v2 mixed dataset."""

    from ultralytics import YOLO

    if name is None:
        name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_arg = str(Path(model).resolve()) if Path(model).exists() else model
    yolo = YOLO(model_arg)
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml.resolve()),
        "epochs": int(epochs),
        "patience": int(patience),
        "batch": int(batch),
        "imgsz": int(imgsz),
        "project": str(project.resolve()),
        "name": name,
        "exist_ok": bool(exist_ok),
        "device": device,
        "workers": int(workers),
        "seed": int(seed),
        "verbose": True,
    }
    if Path(model).suffix.lower() in {".yaml", ".yml"}:
        train_kwargs["pretrained"] = False
    yolo.train(**train_kwargs)

    run_dir = project / name
    best_model = run_dir / "weights" / "best.pt"
    report: dict[str, Any] = {
        "scheme": "DRFT-v2 detector evaluation",
        "data_yaml": str(data_yaml.resolve()),
        "model": model_arg,
        "run_dir": str(run_dir.resolve()),
        "best_model": str(best_model.resolve()),
        "train": {
            "epochs": int(epochs),
            "patience": int(patience),
            "batch": int(batch),
            "imgsz": int(imgsz),
            "device": device,
            "workers": int(workers),
            "seed": int(seed),
        },
    }
    if test_after_train and best_model.exists():
        metrics = YOLO(str(best_model)).val(
            data=str(data_yaml.resolve()),
            split="test",
            project=str(run_dir.resolve()),
            name="test_results",
            exist_ok=True,
            workers=int(workers),
            device=device,
            verbose=True,
        )
        report["test_metrics"] = _metrics_to_dict(metrics)
    _write_json(run_dir / "experiment_report.json", report)
    console.print(f"[green]YOLO run saved to {run_dir}[/green]")


if __name__ == "__main__":
    app()
