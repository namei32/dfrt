from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neu_det_pipeline.data import collect_dataset_instances
from neu_det_pipeline.models.reference_cmdp import (
    bbox_inside_crop,
    build_box_adaptive_pixel_masks,
    build_dbt_pixel_masks,
    build_scmp_pixel_masks,
    estimate_pseudo_defect_mask,
    expand_bbox,
    infer_surface_domain,
    propagate_background_latents,
    reference_caption,
    scale_bbox_to_size,
    sample_stochastic_box_params,
    shape_preserving_pixel_blend,
    transition_eta,
)


def test_reference_scmp_mask_uses_context_outside_defect_box() -> None:
    context, defect = build_scmp_pixel_masks((64, 64), (20, 18, 42, 40), feather_radius=0.0)

    context_arr = np.asarray(context, dtype=np.uint8)
    defect_arr = np.asarray(defect, dtype=np.uint8)

    assert defect_arr[24:34, 24:34].min() == 255
    assert context_arr[24:34, 24:34].max() == 0
    assert context_arr[:10, :10].min() == 255
    assert defect_arr[:10, :10].max() == 0


def test_reference_expanded_bbox_and_local_bbox_are_clipped() -> None:
    expanded = expand_bbox((2, 4, 20, 16), (32, 24), dilation=2.0)
    local = bbox_inside_crop((2, 4, 20, 16), expanded)
    scaled = scale_bbox_to_size(local, (expanded[2] - expanded[0], expanded[3] - expanded[1]), (64, 64))

    assert expanded[0] == 0
    assert expanded[1] <= 4
    assert local[0] >= 0
    assert local[1] >= 0
    assert 0 <= scaled[0] < scaled[2] <= 64
    assert 0 <= scaled[1] < scaled[3] <= 64


def test_reference_domain_inference_prefers_fabric_keywords() -> None:
    assert infer_surface_domain(Path("data/TILDA/raw"), "auto") == "fabric"
    assert infer_surface_domain(Path("data/NEU/split"), "auto") == "steel"
    assert infer_surface_domain(Path("anything"), "fabric") == "fabric"


def test_reference_caption_is_class_aware_for_gc10_defects() -> None:
    prompt = reference_caption("steel", "oil_spot")

    assert "hot-rolled steel" in prompt
    assert "oil spot" in prompt
    assert prompt != reference_caption("steel")


def test_split_voc_layout_is_collected_for_reference_pipeline(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    ann_dir = tmp_path / "annotations" / "train"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(image_dir / "sample.jpg")
    (ann_dir / "sample.xml").write_text(
        """
<annotation>
  <object><name>crazing</name><bndbox><xmin>3</xmin><ymin>4</ymin><xmax>20</xmax><ymax>24</ymax></bndbox></object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    samples = collect_dataset_instances(tmp_path)

    assert len(samples) == 1
    assert samples[0].target_key == "sample_o00"
    assert samples[0].bbox == (3, 4, 20, 24)
    assert samples[0].image_path == image_dir / "sample.jpg"


def test_dbt_pseudo_mask_is_estimated_inside_bbox() -> None:
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    arr[26:39, 23:42] = 55
    image = Image.fromarray(arr, mode="RGB")

    pseudo = estimate_pseudo_defect_mask(image, (18, 18, 48, 48), blur_sigma=2.0)
    pseudo_arr = np.asarray(pseudo, dtype=np.uint8)

    assert pseudo_arr[18:48, 18:48].max() == 255
    assert pseudo_arr[:12, :].max() == 0
    assert pseudo_arr[:, :12].max() == 0


def test_dbt_trimasks_partition_the_crop() -> None:
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    arr[24:40, 24:40] = 40
    masks = build_dbt_pixel_masks(
        Image.fromarray(arr, mode="RGB"),
        (18, 18, 48, 48),
        core_radius=1,
        band_radius=3,
        min_area_ratio=0.02,
    )

    core = np.asarray(masks.core, dtype=np.uint8) > 127
    band = np.asarray(masks.band, dtype=np.uint8) > 127
    out = np.asarray(masks.out, dtype=np.uint8) > 127

    assert masks.stats["pseudo_area"] > 0
    assert not np.logical_and(core, band).any()
    assert not np.logical_and(core, out).any()
    assert not np.logical_and(band, out).any()
    assert np.logical_or.reduce((core, band, out)).all()


def test_box_adaptive_masks_use_bbox_as_edit_domain() -> None:
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    arr[25:38, 24:43] = 45
    bbox = (18, 18, 48, 48)
    masks = build_box_adaptive_pixel_masks(
        Image.fromarray(arr, mode="RGB"),
        bbox,
        band_radius=4,
        min_area_ratio=0.02,
        max_area_ratio=0.30,
    )

    core = np.asarray(masks.core, dtype=np.uint8) > 127
    band = np.asarray(masks.band, dtype=np.uint8) > 127
    out = np.asarray(masks.out, dtype=np.uint8) > 127
    defect = np.asarray(masks.defect, dtype=np.uint8) > 127
    pseudo = np.asarray(masks.pseudo, dtype=np.uint8) > 127

    assert masks.stats["mask_geometry"] == "box-adaptive"
    assert defect[bbox[1] + 2 : bbox[3] - 2, bbox[0] + 2 : bbox[2] - 2].any()
    assert not defect[: bbox[1] - 2, :].any()
    assert pseudo[defect].any()
    assert not pseudo[~defect].any()
    assert not np.logical_and(core, band).any()
    assert not np.logical_and(core, out).any()
    assert not np.logical_and(band, out).any()
    assert np.logical_or.reduce((core, band, out)).all()


def test_stochastic_box_params_are_seeded_and_class_aware() -> None:
    first = sample_stochastic_box_params("oil_spot", "sample_o00", index=3, seed=42)
    second = sample_stochastic_box_params("oil_spot", "sample_o00", index=3, seed=42)
    linear = sample_stochastic_box_params("crease", "sample_o00", index=3, seed=42)

    assert first == second
    assert first.family == "blob"
    assert linear.family == "linear"
    assert 6 <= first.band_radius <= 12
    assert 0.24 <= first.max_area_ratio <= 0.38
    assert 0.35 <= first.box_boundary_strength <= 0.55


def test_shape_preserving_pixel_blend_limits_residual_to_mask_shape() -> None:
    original = Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8), mode="RGB")
    generated = Image.fromarray(np.full((16, 16, 3), [200, 40, 160], dtype=np.uint8), mode="RGB")
    core = Image.new("L", (16, 16), 0)
    core_arr = np.asarray(core, dtype=np.uint8).copy()
    core_arr[4:12, 4:12] = 255
    core = Image.fromarray(core_arr, mode="L")
    band = Image.new("L", (16, 16), 0)

    blended = shape_preserving_pixel_blend(
        original,
        generated,
        core,
        band,
        core_strength=1.0,
        band_strength=0.0,
        residual_clip=20.0,
        luminance_only=True,
    )
    arr = np.asarray(blended, dtype=np.int16)

    assert np.all(arr[:3, :3] == 100)
    assert arr[8, 8, 0] == arr[8, 8, 1] == arr[8, 8, 2]
    assert abs(int(arr[8, 8, 0]) - 100) <= 20


def test_shape_preserving_pixel_blend_dampens_contrast_sign_flip() -> None:
    arr = np.full((21, 21, 3), 100, dtype=np.uint8)
    arr[8:13, 8:13] = 165
    original = Image.fromarray(arr, mode="RGB")
    generated_arr = arr.copy()
    generated_arr[8:13, 8:13] = 40
    generated = Image.fromarray(generated_arr, mode="RGB")
    core = Image.fromarray(np.full((21, 21), 255, dtype=np.uint8), mode="L")
    band = Image.new("L", (21, 21), 0)

    blended = shape_preserving_pixel_blend(
        original,
        generated,
        core,
        band,
        core_strength=1.0,
        contrast_preservation=1.0,
        contrast_blur_sigma=2.0,
        contrast_min=1.0,
    )
    arr = np.asarray(blended, dtype=np.uint8)

    assert arr[10, 10, 0] == 165


def test_no_leak_background_propagation_uses_only_out_mask() -> None:
    latents = torch.full((1, 1, 9, 9), 9.0)
    latents[:, :, :, :3] = 1.0
    out_mask = torch.zeros((1, 1, 9, 9))
    out_mask[:, :, :, :3] = 1.0

    propagated = propagate_background_latents(latents, out_mask, sigma=1.2)

    assert float(propagated.max()) <= 1.001
    assert float(propagated[:, :, 4, 4]) > 0.0


def test_transition_eta_is_larger_at_noisier_timesteps() -> None:
    class Scheduler:
        alphas_cumprod = torch.linspace(1.0, 0.001, 1000)

    clean_eta = transition_eta(
        Scheduler(),
        torch.tensor([0]),
        eta_min=0.3,
        eta_max=0.85,
        step_idx=49,
        num_steps=50,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    noisy_eta = transition_eta(
        Scheduler(),
        torch.tensor([999]),
        eta_min=0.3,
        eta_max=0.85,
        step_idx=0,
        num_steps=50,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert float(noisy_eta) > float(clean_eta)
