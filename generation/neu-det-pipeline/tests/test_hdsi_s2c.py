from __future__ import annotations

import numpy as np
from PIL import Image

from neu_det_pipeline.models.context_lgi import (
    DELGIConfig,
    _annotate_hdsi_target_spectrum,
    _fallback_spectrum_observation,
    _hdsi_s2c_consistency,
    _hdsi_s2c_quality,
    _spectrum_observation_to_dict,
    _spectrum_profile_from_observations,
)


def test_hdsi_s2c_consistency_extracts_generated_spectrum() -> None:
    source = np.full((64, 64), 128, dtype=np.uint8)
    generated = source.copy()
    generated[28:36, 12:52] = 176
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[26:38, 10:54] = 255

    metrics = _hdsi_s2c_consistency(
        Image.fromarray(source, mode="L").convert("RGB"),
        Image.fromarray(generated, mode="L").convert("RGB"),
        Image.fromarray(mask, mode="L"),
        Image.fromarray(mask, mode="L"),
        target_spectrum={
            "low_freq_ratio": 0.28,
            "mid_freq_ratio": 0.34,
            "high_freq_ratio": 0.38,
            "orientation_0": 0.70,
            "orientation_45": 0.10,
            "orientation_90": 0.10,
            "orientation_135": 0.10,
            "orientation_bin": 0,
            "boundary_roughness": 0.60,
            "component_density": 0.70,
            "area_fraction": 0.18,
            "contrast_ratio": 1.50,
        },
        srt_plan=None,
        s2i_metrics={"s2i_score": 0.72},
        config=DELGIConfig(use_hdsi_s2c=True, hdsi_s2c_min_score=0.05),
    )

    generated_spectrum = metrics["hdsi_s2c_generated_spectrum"]
    assert isinstance(generated_spectrum, dict)
    assert generated_spectrum["valid"] is True
    assert 0.0 <= metrics["hdsi_s2c_score"] <= 1.0
    assert 0.0 <= metrics["hdsi_s2c_spectrum_match"] <= 1.0


def test_hdsi_s2c_quality_rewards_high_consistency() -> None:
    base_quality = {
        "inside_delta": 0.022,
        "outside_delta": 0.010,
        "source_similarity": 0.958,
        "s2i_score": 0.70,
        "s2i_ratio": 1.40,
        "hdsi_intervention_score": 0.72,
        "hdsi_validity_score": 0.82,
        "source_spectrum_distance": 0.34,
        "srt_de_lgi_total": 0.48,
        "de_lgi_total": 0.48,
    }
    high = _hdsi_s2c_quality(
        dict(base_quality),
        cls_name="patches",
        s2c_metrics={
            "hdsi_s2c_score": 0.82,
            "hdsi_s2c_structure_score": 0.78,
            "hdsi_s2c_image_score": 0.84,
            "hdsi_s2c_spectrum_match": 0.80,
            "hdsi_s2c_visible_score": 0.70,
            "hdsi_s2c_gate_failures": [],
        },
        config=DELGIConfig(use_hdsi_s2c=True),
    )
    low = _hdsi_s2c_quality(
        dict(base_quality),
        cls_name="patches",
        s2c_metrics={
            "hdsi_s2c_score": 0.18,
            "hdsi_s2c_structure_score": 0.20,
            "hdsi_s2c_image_score": 0.15,
            "hdsi_s2c_spectrum_match": 0.12,
            "hdsi_s2c_visible_score": 0.25,
            "hdsi_s2c_gate_failures": ["image_spectrum_mismatch"],
        },
        config=DELGIConfig(use_hdsi_s2c=True),
    )

    assert high["hdsi_s2c_gate_pass"] is True
    assert low["hdsi_s2c_gate_pass"] is False
    assert high["hdsi_s2c_selection_score"] > low["hdsi_s2c_selection_score"]


def test_hdsi_target_records_validity_and_source_distance_gates() -> None:
    obs = _fallback_spectrum_observation("scratch")
    profile = _spectrum_profile_from_observations("scratch", [obs])
    source = _spectrum_observation_to_dict(obs)
    candidate = {
        **source,
        "area_fraction": 0.92,
        "contrast_ratio": 18.0,
        "elongation": 11.0,
    }

    annotated = _annotate_hdsi_target_spectrum(
        candidate,
        profile=profile,
        source_spectrum=source,
        history=[],
        coverage_counts=None,
        config=DELGIConfig(hdsi_min_validity_score=0.95, hdsi_min_source_distance=0.95),
    )

    assert annotated["hdsi_validity_gate_pass"] is False
    assert annotated["hdsi_source_distance_gate_pass"] is False
    assert annotated["hdsi_hard_valid_gate_pass"] is False
    assert annotated["hdsi_validity_floor"] == 0.95
    assert annotated["hdsi_source_distance_floor"] == 0.95
