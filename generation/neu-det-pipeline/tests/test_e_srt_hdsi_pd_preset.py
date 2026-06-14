from __future__ import annotations

from neu_det_pipeline.data.resplit import _candidate_annotation_gates_pass
from neu_det_pipeline.models.context_lgi import DELGIConfig, _e_srt_hdsi_pd_quality


def test_non_strict_s2i_is_rank_only_for_e_srt_hdsi_pd_rows() -> None:
    row = {
        "candidate_accepted": True,
        "method": "e_srt_hdsi_pd_de_lgi",
        "label_contract": "source-label-inherited",
        "quality": {
            "s2i_gate_pass": False,
            "seca_gate_pass": True,
        },
        "latent_inpainting": {
            "srt_strict_s2i_gate": False,
        },
    }

    assert _candidate_annotation_gates_pass(row) is True


def test_strict_s2i_still_rejects_failed_structure_rows() -> None:
    row = {
        "candidate_accepted": True,
        "method": "spec_srt_lgi",
        "label_contract": "source-label-inherited",
        "quality": {
            "s2i_gate_pass": False,
            "seca_gate_pass": True,
        },
        "latent_inpainting": {
            "srt_strict_s2i_gate": True,
        },
    }

    assert _candidate_annotation_gates_pass(row) is False


def test_e_srt_hdsi_pd_quality_rejects_copy_like_candidates() -> None:
    quality = {
        "inside_delta": 0.006,
        "outside_delta": 0.010,
        "source_similarity": 0.999,
        "visible_change_fraction": 0.004,
        "s2i_score": 0.20,
        "s2i_ratio": 0.40,
        "hdsi_pd_selection_score": 0.65,
        "srt_de_lgi_total": 0.45,
    }

    scored = _e_srt_hdsi_pd_quality(quality, cls_name="inclusion", config=DELGIConfig())

    assert scored["e_srt_hdsi_pd_gate_pass"] is False
    assert "source_copy" in scored["e_srt_hdsi_pd_gate_failures"]
    assert scored["e_srt_hdsi_pd_selection_score"] < 0.45


def test_e_srt_hdsi_pd_quality_keeps_visible_structural_candidates() -> None:
    quality = {
        "inside_delta": 0.021,
        "outside_delta": 0.012,
        "source_similarity": 0.955,
        "visible_change_fraction": 0.12,
        "s2i_score": 0.66,
        "s2i_ratio": 1.32,
        "hdsi_pd_selection_score": 0.68,
        "srt_de_lgi_total": 0.48,
    }

    scored = _e_srt_hdsi_pd_quality(quality, cls_name="patches", config=DELGIConfig())

    assert scored["e_srt_hdsi_pd_gate_pass"] is True
    assert scored["e_srt_hdsi_pd_gate_failures"] == []
    assert scored["e_srt_hdsi_pd_selection_score"] > quality["srt_de_lgi_total"]


def test_e_srt_hdsi_pd_quality_rejects_low_novelty_near_source_rows() -> None:
    quality = {
        "inside_delta": 0.020,
        "outside_delta": 0.010,
        "source_similarity": 0.988,
        "visible_change_fraction": 0.09,
        "s2i_score": 0.61,
        "s2i_ratio": 1.21,
        "hdsi_pd_selection_score": 0.70,
        "structure_novelty_score": 0.02,
        "residual_recomposition_score": 0.03,
        "diversity_gain_score": 0.04,
        "srt_de_lgi_total": 0.62,
    }

    scored = _e_srt_hdsi_pd_quality(quality, cls_name="crazing", config=DELGIConfig())

    assert scored["e_srt_hdsi_pd_gate_pass"] is False
    assert "low_novelty" in scored["e_srt_hdsi_pd_gate_failures"]
    assert scored["e_srt_hdsi_pd_novelty_score"] < 0.24


def test_e_srt_hdsi_pd_quality_rewards_structural_and_spectral_novelty() -> None:
    common = {
        "inside_delta": 0.020,
        "outside_delta": 0.010,
        "source_similarity": 0.970,
        "visible_change_fraction": 0.09,
        "s2i_score": 0.61,
        "s2i_ratio": 1.21,
        "hdsi_pd_selection_score": 0.70,
        "srt_de_lgi_total": 0.50,
    }
    low = _e_srt_hdsi_pd_quality(
        {**common, "structure_novelty_score": 0.05, "residual_recomposition_score": 0.08, "diversity_gain_score": 0.06},
        cls_name="rolled-in_scale",
        config=DELGIConfig(),
    )
    high = _e_srt_hdsi_pd_quality(
        {**common, "structure_novelty_score": 0.62, "residual_recomposition_score": 0.70, "diversity_gain_score": 0.58},
        cls_name="rolled-in_scale",
        config=DELGIConfig(),
    )

    assert high["e_srt_hdsi_pd_gate_pass"] is True
    assert high["e_srt_hdsi_pd_novelty_score"] > low["e_srt_hdsi_pd_novelty_score"]
    assert high["e_srt_hdsi_pd_selection_score"] > low["e_srt_hdsi_pd_selection_score"]
