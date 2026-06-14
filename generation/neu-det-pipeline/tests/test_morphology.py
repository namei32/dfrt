from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neu_det_pipeline.data.loader import DefectSample
from neu_det_pipeline.guidance.morphology import MorphologyPrior, build_morphology_calibrated_mask


def _sample_image(path: Path, cls_name: str = "inclusion") -> DefectSample:
    image = Image.new("RGB", (128, 128), (120, 120, 120))
    draw = ImageDraw.Draw(image)
    draw.ellipse([46, 48, 84, 78], fill=(178, 178, 178))
    if cls_name == "scratches":
        draw.line([(28, 72), (102, 59)], fill=(190, 190, 190), width=4)
    image.save(path)
    return DefectSample(path, path.with_suffix(".xml"), cls_name, (40, 42, 92, 84))


def test_morphology_calibrated_mask_stays_inside_target_bbox(tmp_path: Path) -> None:
    sample = _sample_image(tmp_path / "inclusion_1.jpg")
    prior = MorphologyPrior.from_samples([sample])

    mask, plan = build_morphology_calibrated_mask(
        prior,
        "inclusion",
        sample.bbox,
        target_key="target",
        target_size=(128, 128),
        target_source_size=(128, 128),
        seed=11,
        feather_radius=0.0,
    )

    arr = np.asarray(mask, dtype=np.uint8) > 0
    assert arr.any()
    ys, xs = np.where(arr)
    assert xs.min() >= sample.bbox[0]
    assert xs.max() < sample.bbox[2] + 1
    assert ys.min() >= sample.bbox[1]
    assert ys.max() < sample.bbox[3] + 1
    assert plan.profile_key == sample.image_path.stem
    assert plan.actual_coverage > 0.0


def test_morphology_prior_falls_back_for_unseen_class() -> None:
    prior = MorphologyPrior({})
    mask, plan = build_morphology_calibrated_mask(
        prior,
        "scratches",
        (20, 24, 96, 80),
        target_size=(128, 128),
        target_source_size=(128, 128),
        seed=3,
        feather_radius=0.0,
    )

    assert np.asarray(mask, dtype=np.uint8).max() > 0
    assert plan.profile_key == "scratches_fallback"
    assert plan.family == "scratch"
