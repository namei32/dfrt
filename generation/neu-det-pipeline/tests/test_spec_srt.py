from __future__ import annotations

import numpy as np
from PIL import Image

from neu_det_pipeline.models.context_lgi import (
    DELGIConfig,
    LGIMaskSet,
    _quality_domains,
    _spec_srt_quality,
)


def _image(values: np.ndarray) -> Image.Image:
    rgb = np.repeat(values[..., None], 3, axis=2)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _mask_set(size: int = 32) -> LGIMaskSet:
    core = np.zeros((size, size), dtype=np.uint8)
    core[8:24, 8:24] = 255
    shell = np.zeros_like(core)
    shell[6:26, 6:26] = 255
    shell[8:24, 8:24] = 0
    out = np.full_like(core, 255)
    out[6:26, 6:26] = 0
    return LGIMaskSet(
        core=Image.fromarray(core, mode="L"),
        shell=Image.fromarray(shell, mode="L"),
        out=Image.fromarray(out, mode="L"),
        edit=Image.fromarray(np.maximum(core, shell), mode="L"),
        stats={},
    )


def test_quality_domains_tracks_visible_change_and_copy_similarity() -> None:
    base = np.full((32, 32), 126, dtype=np.float32)
    yy, xx = np.indices((16, 16))
    texture = ((xx + yy) % 2) * 55
    source = base.copy()
    generated = base.copy()
    source[8:24, 8:24] = 95 + texture
    generated[8:24, 8:24] = 145 - texture

    quality = _quality_domains(_image(source), _image(generated), _mask_set())

    assert quality["inside_delta"] > 0.10
    assert quality["visible_change_fraction"] > 0.90
    assert quality["source_similarity"] < 0.70
    assert quality["sharpness_ratio"] > 0.75


def test_spec_srt_quality_rejects_copy_like_candidates() -> None:
    config = DELGIConfig(
        method_preset="spec-srt",
        use_ucs=True,
        use_srt=True,
        use_hdsi=True,
        use_e_srt=True,
        spec_srt_min_inside_delta=0.018,
        spec_srt_min_change_fraction=0.035,
        spec_srt_max_source_similarity=0.992,
        spec_srt_min_sharpness_ratio=0.62,
    )
    base_quality = {
        "total": 0.55,
        "background_score": 0.85,
        "defect_strength_score": 0.65,
        "boundary_score": 0.70,
        "inside_delta": 0.024,
        "boundary_delta": 0.012,
        "outside_delta": 0.010,
        "visible_change_fraction": 0.08,
        "source_similarity": 0.985,
        "sharpness_ratio": 0.82,
    }
    passed = _spec_srt_quality(
        base_quality,
        energy_target=0.22,
        energy_actual=0.21,
        background_leakage=0.01,
        spectrum_match=0.58,
        utility_score=0.62,
        diversity_gain=0.70,
        structure_transport_score=0.55,
        hdsi_intervention_score=0.45,
        hdsi_validity_score=0.90,
        source_spectrum_distance=0.36,
        s2i_score=0.66,
        s2i_visible_delta=0.024,
        s2i_ratio=1.60,
        s2i_gate_pass=True,
        config=config,
    )
    copied = _spec_srt_quality(
        {**base_quality, "inside_delta": 0.012, "visible_change_fraction": 0.01, "source_similarity": 0.998},
        energy_target=0.22,
        energy_actual=0.21,
        background_leakage=0.01,
        spectrum_match=0.58,
        utility_score=0.62,
        diversity_gain=0.70,
        structure_transport_score=0.55,
        hdsi_intervention_score=0.45,
        hdsi_validity_score=0.90,
        source_spectrum_distance=0.36,
        s2i_score=0.66,
        s2i_visible_delta=0.024,
        s2i_ratio=1.60,
        s2i_gate_pass=True,
        config=config,
    )

    assert passed["spec_srt_gate_pass"] is True
    assert passed["source_spectrum_distance"] == 0.36
    assert passed["spec_srt_gate_failures"] == []
    assert passed["spec_srt_gate_thresholds"]["inside_delta"] == 0.018
    assert passed["spec_srt_gate_margins"]["source_spectrum_distance"] > 0
    assert copied["spec_srt_gate_pass"] is False
    assert "source_copy" in copied["spec_srt_gate_failures"]
    assert copied["spec_srt_selection_score"] < passed["spec_srt_selection_score"]


def test_spec_srt_quality_uses_family_normalized_visibility_gate() -> None:
    config = DELGIConfig(
        method_preset="spec-srt",
        use_ucs=True,
        use_srt=True,
        use_hdsi=True,
        use_e_srt=True,
        spec_srt_min_inside_delta=0.018,
        spec_srt_min_change_fraction=0.035,
        spec_srt_max_source_similarity=0.985,
        spec_srt_min_sharpness_ratio=0.50,
    )
    thin_scratch_quality = {
        "total": 0.52,
        "background_score": 0.92,
        "defect_strength_score": 0.45,
        "boundary_score": 0.72,
        "inside_delta": 0.0125,
        "boundary_delta": 0.010,
        "outside_delta": 0.009,
        "visible_change_fraction": 0.07,
        "source_similarity": 0.997,
        "sharpness_ratio": 0.80,
    }

    generic = _spec_srt_quality(
        thin_scratch_quality,
        cls_name="inclusion",
        energy_target=0.20,
        energy_actual=0.19,
        background_leakage=0.01,
        spectrum_match=0.50,
        utility_score=0.55,
        diversity_gain=0.60,
        structure_transport_score=0.50,
        hdsi_intervention_score=0.45,
        hdsi_validity_score=0.90,
        source_spectrum_distance=0.30,
        s2i_score=0.60,
        s2i_visible_delta=0.017,
        s2i_ratio=1.05,
        s2i_gate_pass=True,
        config=config,
    )
    scratch = _spec_srt_quality(
        thin_scratch_quality,
        cls_name="scratches",
        energy_target=0.20,
        energy_actual=0.19,
        background_leakage=0.01,
        spectrum_match=0.50,
        utility_score=0.55,
        diversity_gain=0.60,
        structure_transport_score=0.50,
        hdsi_intervention_score=0.45,
        hdsi_validity_score=0.90,
        source_spectrum_distance=0.30,
        s2i_score=0.60,
        s2i_visible_delta=0.017,
        s2i_ratio=1.05,
        s2i_gate_pass=True,
        config=config,
    )

    assert generic["spec_srt_gate_pass"] is False
    assert scratch["spec_srt_gate_pass"] is True
    assert scratch["spec_srt_gate_thresholds"]["inside_delta"] < generic["spec_srt_gate_thresholds"]["inside_delta"]
