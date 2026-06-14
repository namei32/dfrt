from __future__ import annotations

import torch

from neu_det_pipeline.models.context_lgi import (
    DELGIConfig,
    _apply_hdsi_pd_detail_refinement,
    _compose_hdsi_pd_prototype_delta,
    _latent_highpass,
    _masked_rms_tensor,
)


def _target_spectrum() -> dict[str, float | int]:
    return {
        "low_freq_ratio": 0.18,
        "mid_freq_ratio": 0.32,
        "high_freq_ratio": 0.50,
        "orientation_0": 0.10,
        "orientation_45": 0.55,
        "orientation_90": 0.20,
        "orientation_135": 0.15,
        "orientation_bin": 1,
        "polarity": -0.15,
        "boundary_roughness": 1.10,
        "component_density": 1.20,
    }


def _core_and_shell() -> tuple[torch.Tensor, torch.Tensor]:
    core = torch.zeros(1, 1, 8, 8)
    core[:, :, 2:6, 2:6] = 1.0
    shell = torch.zeros_like(core)
    shell[:, :, 1:7, 1:7] = 1.0
    shell = (shell - core).clamp(0.0, 1.0)
    return core, shell


def test_hdsi_pd_composes_nonzero_prototype_delta() -> None:
    torch.manual_seed(7)
    core, _shell = _core_and_shell()
    deltas = [
        torch.randn(1, 4, 8, 8) * 0.13,
        torch.randn(1, 4, 8, 8) * 0.09,
    ]
    config = DELGIConfig(use_hdsi=True, use_hdsi_pd=True, hdsi_pd_strength=0.45)

    prototype, info = _compose_hdsi_pd_prototype_delta(
        deltas,
        core,
        target_energy=0.22,
        target_spectrum=_target_spectrum(),
        seed=11,
        config=config,
    )

    assert prototype is not None
    assert prototype.shape == deltas[0].shape
    assert info["hdsi_pd_enabled"] is True
    assert info["hdsi_pd_phase_operator"] == "fft_unit_phase_blend"
    assert float(_masked_rms_tensor(_latent_highpass(prototype), core)) > 0.01


def test_hdsi_pd_late_detail_gate_changes_only_after_start() -> None:
    torch.manual_seed(13)
    core, shell = _core_and_shell()
    base = torch.zeros(1, 4, 8, 8)
    prototype = torch.randn_like(base) * 0.11
    noise = torch.randn_like(base)
    config = DELGIConfig(
        use_hdsi=True,
        use_hdsi_pd=True,
        hdsi_pd_late_detail_strength=0.32,
        hdsi_pd_detail_start=0.60,
        hdsi_pd_late_renoise_strength=0.05,
    )

    early, early_trace = _apply_hdsi_pd_detail_refinement(
        base,
        prototype,
        core,
        shell,
        target_energy=0.20,
        target_spectrum=_target_spectrum(),
        step_fraction=0.30,
        config=config,
        detail_noise=noise,
    )
    late, late_trace = _apply_hdsi_pd_detail_refinement(
        base,
        prototype,
        core,
        shell,
        target_energy=0.20,
        target_spectrum=_target_spectrum(),
        step_fraction=0.90,
        config=config,
        detail_noise=noise,
    )

    assert torch.allclose(early, base)
    assert early_trace["hdsi_pd_detail_gate"] == 0.0
    assert not torch.allclose(late, base)
    assert late_trace["hdsi_pd_detail_gate"] > 0.0


def test_hdsi_pd_late_detail_is_core_only() -> None:
    torch.manual_seed(17)
    core, shell = _core_and_shell()
    base = torch.zeros(1, 4, 8, 8)
    prototype = torch.randn_like(base) * 0.11
    noise = torch.randn_like(base)
    config = DELGIConfig(
        use_hdsi=True,
        use_hdsi_pd=True,
        hdsi_pd_late_detail_strength=0.36,
        hdsi_pd_detail_start=0.40,
        hdsi_pd_late_renoise_strength=0.07,
    )

    late, trace = _apply_hdsi_pd_detail_refinement(
        base,
        prototype,
        core,
        shell,
        target_energy=0.20,
        target_spectrum=_target_spectrum(),
        step_fraction=0.90,
        config=config,
        detail_noise=noise,
    )

    outside = (1.0 - (core + shell).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    assert trace["hdsi_pd_detail_gate"] > 0.0
    assert float(((late - base).abs() * core).max()) > 0.0
    assert torch.allclose((late - base) * shell, torch.zeros_like(late), atol=1e-7)
    assert torch.allclose((late - base) * outside, torch.zeros_like(late), atol=1e-7)
