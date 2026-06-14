from __future__ import annotations

import numpy as np
from PIL import Image

from neu_det_pipeline.data import collect_dataset_images, collect_dataset_instances
from neu_det_pipeline.data.resplit import create_mixed_dataset, source_stem_for_generated
from neu_det_pipeline.guidance.drft import (
    build_adaptive_counterfactual_canvas,
    build_class_aware_defect_residual_field,
    build_context_preserved_drft_contract,
    build_residual_seeded_canvas,
    score_drft_context_contract,
)


def test_adaptive_counterfactual_canvas_returns_v2_metadata_and_mask() -> None:
    base = np.full((128, 128, 3), 132, dtype=np.float32)
    yy, xx = np.mgrid[:128, :128]
    del yy
    base += (8.0 * np.sin(xx / 5.0))[:, :, None]
    base[30:94, 60:64] = 230
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")

    clean, erase, meta = build_adaptive_counterfactual_canvas(
        image,
        (48, 20, 78, 104),
        "scratches",
        seed=7,
        candidates=3,
    )

    assert clean.size == image.size
    assert erase.size == image.size
    assert meta["variant"] == "drft-v2"
    assert np.asarray(erase).max() > 0


def test_class_aware_residual_field_has_valid_channels() -> None:
    clean = Image.fromarray(np.full((128, 128, 3), 128, dtype=np.uint8), mode="RGB")
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[28:100, 58:66] = 255

    field = build_class_aware_defect_residual_field(
        clean,
        Image.fromarray(mask, mode="L"),
        "scratches",
        orientation_deg=90.0,
        seed=11,
    )

    assert field.soft_mask.shape == (128, 128)
    assert field.signed_residual.shape == (128, 128)
    assert field.boundary_shell.shape == (128, 128)
    assert np.isfinite(field.signed_residual).all()
    assert float(np.abs(field.signed_residual).max()) > 1.0


def test_context_contract_preserves_protected_instance_area() -> None:
    clean = Image.fromarray(np.full((128, 128, 3), 128, dtype=np.uint8), mode="RGB")
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[34:94, 48:76] = 255
    protect = np.zeros((128, 128), dtype=np.uint8)
    protect[58:74, 64:98] = 255
    field = build_class_aware_defect_residual_field(
        clean,
        Image.fromarray(mask, mode="L"),
        "scratches",
        orientation_deg=90.0,
        seed=13,
    )

    contract = build_context_preserved_drft_contract(
        Image.fromarray(mask, mode="L"),
        field,
        (48, 34, 76, 94),
        (128, 128),
        protect_mask=Image.fromarray(protect, mode="L"),
        context_dilation=1.7,
        shell_weight=0.6,
    )

    edit = np.asarray(contract.edit_mask)
    context = np.asarray(contract.context_mask)
    assert edit.max() > 0
    assert edit.max() >= 240
    assert context.max() > 0
    assert edit[58:74, 64:98].max() == 0
    assert contract.stats["variant"] == "drft-v2-context"
    assert contract.stats["edit_within_expanded_bbox"] is True


def test_context_contract_quality_rewards_preserved_context() -> None:
    original = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8), mode="RGB")
    generated = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8), mode="RGB")
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 255
    field = build_class_aware_defect_residual_field(
        original,
        Image.fromarray(mask, mode="L"),
        "patches",
        seed=3,
    )
    contract = build_context_preserved_drft_contract(
        Image.fromarray(mask, mode="L"),
        field,
        (24, 24, 40, 40),
        (64, 64),
    )

    quality = score_drft_context_contract(original, generated, contract)

    assert quality.context_preservation > 0.99
    assert quality.background_preservation > 0.99
    assert quality.total > 0.99


def test_residual_seeded_canvas_changes_core_evidence() -> None:
    clean = Image.fromarray(np.full((96, 96, 3), 128, dtype=np.uint8), mode="RGB")
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[30:66, 40:56] = 255
    field = build_class_aware_defect_residual_field(
        clean,
        Image.fromarray(mask, mode="L"),
        "scratches",
        orientation_deg=90.0,
        seed=17,
    )

    seeded, meta = build_residual_seeded_canvas(clean, field, residual_seed_gain=1.2)
    diff = np.abs(np.asarray(seeded, dtype=np.float32) - np.asarray(clean, dtype=np.float32)).mean(axis=2)

    assert meta["enabled"] is True
    assert float(diff[mask > 0].mean()) > 2.0
    assert float(diff[mask == 0].mean()) < float(diff[mask > 0].mean())


def test_residual_seeded_canvas_uses_confidence_guidance() -> None:
    clean = Image.fromarray(np.full((96, 96, 3), 128, dtype=np.uint8), mode="RGB")
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[30:66, 40:56] = 255
    field = build_class_aware_defect_residual_field(
        clean,
        Image.fromarray(mask, mode="L"),
        "scratches",
        orientation_deg=90.0,
        seed=29,
    )
    field.confidence = np.zeros_like(field.soft_mask, dtype=np.float32)
    field.confidence[34:62, 43:53] = 0.95
    field.uncertainty = np.ones_like(field.soft_mask, dtype=np.float32) * 0.2

    seeded, meta = build_residual_seeded_canvas(clean, field, residual_seed_gain=1.2)
    diff = np.abs(np.asarray(seeded, dtype=np.float32) - np.asarray(clean, dtype=np.float32)).mean(axis=2)

    assert meta["enabled"] is True
    assert meta["confidence_guided"] is True
    assert float(diff[34:62, 43:53].mean()) > float(diff[mask > 0].mean()) * 0.45


def test_instance_loader_expands_all_voc_objects(tmp_path) -> None:
    img_dir = tmp_path / "IMAGES"
    ann_dir = tmp_path / "ANNOTATIONS"
    img_dir.mkdir()
    ann_dir.mkdir()
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(img_dir / "sample.jpg")
    (ann_dir / "sample.xml").write_text(
        """
<annotation>
  <object><name>crazing</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>9</xmax><ymax>10</ymax></bndbox></object>
  <object><name>scratches</name><bndbox><xmin>11</xmin><ymin>12</ymin><xmax>19</xmax><ymax>20</ymax></bndbox></object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    samples = collect_dataset_instances(tmp_path)

    assert [sample.target_key for sample in samples] == ["sample_o00", "sample_o01"]
    assert [sample.cls_name for sample in samples] == ["crazing", "scratches"]
    assert all(len(sample.objects) == 2 for sample in samples)


def test_image_loader_keeps_one_target_per_image_with_union_bbox(tmp_path) -> None:
    img_dir = tmp_path / "IMAGES"
    ann_dir = tmp_path / "ANNOTATIONS"
    img_dir.mkdir()
    ann_dir.mkdir()
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(img_dir / "sample.jpg")
    (ann_dir / "sample.xml").write_text(
        """
<annotation>
  <object><name>crazing</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>9</xmax><ymax>10</ymax></bndbox></object>
  <object><name>scratches</name><bndbox><xmin>11</xmin><ymin>12</ymin><xmax>25</xmax><ymax>28</ymax></bndbox></object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    samples = collect_dataset_images(tmp_path)

    assert len(samples) == 1
    assert samples[0].target_key == "sample"
    assert samples[0].bbox == (1, 2, 25, 28)
    assert samples[0].cls_name == "scratches"
    assert len(samples[0].objects) == 2


def test_instance_generated_stem_maps_to_source_stem() -> None:
    assert source_stem_for_generated("crazing_100_o03", {"crazing_100"}) == "crazing_100"
    assert source_stem_for_generated("crazing_100", {"crazing_100"}) == "crazing_100"


def test_mixed_dataset_limits_instance_variants_by_source_and_quality(tmp_path) -> None:
    orig_images = tmp_path / "orig_images"
    orig_labels = tmp_path / "labels"
    gen_images = tmp_path / "generated"
    out_dir = tmp_path / "mixed"
    orig_images.mkdir()
    (orig_labels / "train").mkdir(parents=True)
    gen_images.mkdir()

    Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(orig_images / "sample.jpg")
    (orig_labels / "train" / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text('{"train": ["sample"], "val": [], "test": []}', encoding="utf-8")

    for stem in ["sample_o00", "sample_o01", "sample_o02"]:
        Image.fromarray(np.full((32, 32, 3), 140, dtype=np.uint8)).save(gen_images / f"{stem}.png")
    scores = tmp_path / "candidate_scores.jsonl"
    scores.write_text(
        "\n".join(
            [
                '{"target_key": "sample_o00", "source_key": "sample", "class": "crazing", "selected": true, "quality": {"total": 0.60, "outside_delta": 0.001}, "canvas": {"quality": {"defect_evidence_drop": 0.50}}}',
                '{"target_key": "sample_o01", "source_key": "sample", "class": "crazing", "selected": true, "quality": {"total": 0.90, "outside_delta": 0.001}, "canvas": {"quality": {"defect_evidence_drop": 0.50}}}',
                '{"target_key": "sample_o02", "source_key": "sample", "class": "crazing", "selected": true, "quality": {"total": 0.80, "outside_delta": 0.001}, "canvas": {"quality": {"defect_evidence_drop": 0.50}}}',
            ]
        ),
        encoding="utf-8",
    )

    create_mixed_dataset(
        orig_images_dir=orig_images,
        orig_labels_dir=orig_labels,
        manifest_path=manifest,
        new_images_dir=gen_images,
        run_output_dir=out_dir,
        score_path=scores,
        quality_threshold=0.7,
        per_source_limit=1,
        selection_strategy="quality",
    )

    train_images = sorted(path.name for path in (out_dir / "images" / "train").glob("*"))
    assert train_images == ["sample.jpg", "sample_o01_gen.png"]
