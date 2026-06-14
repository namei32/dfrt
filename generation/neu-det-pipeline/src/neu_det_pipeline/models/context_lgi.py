from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from rich.progress import track

from ..data.loader import DefectSample
from ..guidance.mask import create_composite_preserving_background
from ..guidance.morphology import MorphologyPrior, build_morphology_calibrated_mask
from .reference_cmdp import (
    ReferenceCMDPGenerator,
    bbox_inside_crop,
    expand_bbox,
    mask_to_latent_tensor,
    normalize_latent_triplet,
    propagate_background_latents,
    reference_caption,
    scale_bbox_to_size,
    transition_eta,
)


@dataclass(frozen=True)
class ContextLGIConfig:
    resolution: int = 512
    dilation_factor: float = 1.35
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    seed: int = 42
    candidates_per_sample: int = 1
    eta_min: float = 0.20
    eta_max: float = 0.78
    shell_strength: float = 0.55
    core_strength: float = 1.0
    background_propagation_sigma: float = 1.0
    feather_radius: float = 1.2


@dataclass(frozen=True)
class DELGIConfig:
    method_preset: str = "custom"
    resolution: int = 512
    dilation_factor: float = 1.35
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    seed: int = 42
    candidates_per_sample: int = 1
    eta_min: float = 0.20
    eta_max: float = 0.78
    shell_strength: float = 0.55
    background_leakage: float = 0.03
    background_propagation_sigma: float = 1.0
    boundary_smoothing_sigma: float = 1.15
    feather_radius: float = 1.2
    latent_energy_scale: float = 8.0
    latent_energy_floor: float = 0.12
    latent_energy_ceiling: float = 0.50
    energy_jitter: float = 0.16
    min_projection_scale: float = 0.12
    max_projection_scale: float = 3.40
    use_ucs: bool = True
    spectrum_jitter: float = 0.22
    utility_guidance: float = 0.45
    spectrum_projection_strength: float = 0.82
    spectrum_orientation_strength: float = 0.28
    core_renoise_strength: float = 0.18
    diversity_weight: float = 0.12
    use_drr: bool = False
    residual_bank_mix: float = 0.48
    pseudo_suppression_strength: float = 0.62
    structure_jitter: float = 0.32
    diversity_delta_target: float = 0.070
    outside_delta_budget: float = 0.014
    use_distant_spectrum: bool = False
    distant_spectrum_candidates: int = 8
    distant_spectrum_weight: float = 0.70
    use_hdsi: bool = False
    hdsi_candidates: int = 12
    hdsi_hardness_weight: float = 0.58
    hdsi_validity_weight: float = 0.30
    hdsi_diversity_weight: float = 0.42
    hdsi_projection_boost: float = 0.32
    hdsi_tail_strength: float = 0.55
    hdsi_min_validity_score: float = 0.38
    hdsi_min_source_distance: float = 0.12
    use_hdsi_s2c: bool = False
    hdsi_s2c_structure_weight: float = 0.34
    hdsi_s2c_image_weight: float = 0.46
    hdsi_s2c_min_score: float = 0.42
    hdsi_s2c_strict_gate: bool = True
    hdsi_s2c_spectrum_tolerance: float = 0.34
    use_hdsi_pd: bool = False
    hdsi_pd_prototype_count: int = 2
    hdsi_pd_strength: float = 0.44
    hdsi_pd_phase_strength: float = 0.42
    hdsi_pd_late_detail_strength: float = 0.34
    hdsi_pd_detail_start: float = 0.58
    hdsi_pd_late_renoise_strength: float = 0.10
    use_band_recomposition: bool = False
    band_donor_count: int = 3
    band_recomposition_strength: float = 0.70
    use_srt: bool = False
    srt_transport_strength: float = 0.72
    srt_bbox_jitter: float = 0.30
    srt_scale_jitter: float = 0.34
    srt_boundary_roughness: float = 0.58
    srt_component_jitter: float = 0.55
    srt_source_preservation: float = 0.28
    srt_bbox_update: bool = False
    srt_regeneration_strength: float = 0.42
    srt_regeneration_until: float = 0.45
    srt_late_texture_mix: float = 0.34
    srt_s2i_visible_delta_target: float = 0.018
    srt_s2i_min_ratio: float = 1.35
    use_e_srt: bool = False
    e_srt_evidence_strength: float = 0.34
    e_srt_background_strength: float = 0.42
    e_srt_min_core_energy_ratio: float = 0.72
    e_srt_min_visible_delta: float = 0.012
    e_srt_max_source_similarity: float = 0.992
    e_srt_max_outside_delta: float = 0.025
    e_srt_min_s2i_ratio: float = 0.90
    e_srt_min_novelty_score: float = 0.24
    use_seca: bool = False
    seca_min_coverage: float = 0.18
    seca_max_leakage: float = 0.34
    seca_min_focus: float = 1.18
    seca_strict: bool = True
    srt_strict_s2i_gate: bool = True
    spec_srt_min_inside_delta: float = 0.018
    spec_srt_min_change_fraction: float = 0.035
    spec_srt_max_source_similarity: float = 0.985
    spec_srt_min_sharpness_ratio: float = 0.50
    spec_srt_min_source_spectrum_distance: float = 0.22


@dataclass(frozen=True)
class LGIMaskSet:
    core: Image.Image
    shell: Image.Image
    out: Image.Image
    edit: Image.Image
    stats: dict[str, float]


@dataclass(frozen=True)
class SRTStructurePlan:
    target_key: str
    cls_name: str
    seed: int
    family: str
    source_bbox_scaled: tuple[int, int, int, int]
    transported_bbox_scaled: tuple[int, int, int, int]
    generated_bbox_scaled: tuple[int, int, int, int]
    source_area_fraction: float
    transported_area_fraction: float
    source_local_area_fraction: float
    target_local_area_fraction: float
    transported_local_area_fraction: float
    source_component_count: int
    transported_component_count: int
    source_skeleton_length: float
    transported_skeleton_length: float
    source_mean_width: float
    transported_mean_width: float
    source_boundary_roughness: float
    transported_boundary_roughness: float
    source_orientation_deg: float
    target_orientation_deg: float
    bbox_shift_x: float
    bbox_shift_y: float
    bbox_scale_x: float
    bbox_scale_y: float
    structure_transport_score: float
    hdsi_s2c_structure_score: float
    hdsi_s2c_area_score: float
    hdsi_s2c_orientation_score: float
    hdsi_s2c_roughness_score: float
    hdsi_s2c_density_score: float
    hdsi_s2c_elongation_score: float
    counterfactual_structure_iou: float
    active_bbox_label_coverage: float
    counterfactual_generation: str
    label_source: str


@dataclass(frozen=True)
class DefectEnergyProfile:
    cls_name: str
    sample_count: int
    core_energy_q25: float
    core_energy_q50: float
    core_energy_q75: float
    background_energy_q75: float
    boundary_energy_q50: float
    latent_target_energy: float


class DefectEnergyPrior:
    """Class-wise residual energy statistics mined from existing defect boxes."""

    def __init__(self, profiles: dict[str, DefectEnergyProfile], fallback: DefectEnergyProfile):
        self.profiles = dict(profiles)
        self.fallback = fallback

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[DefectSample],
        *,
        resolution: int,
        dilation_factor: float,
        latent_energy_scale: float,
        latent_energy_floor: float,
        latent_energy_ceiling: float,
    ) -> "DefectEnergyPrior":
        values_by_class: dict[str, list[tuple[float, float, float]]] = {}
        for sample in samples:
            try:
                values = _estimate_sample_energy(sample, resolution=resolution, dilation_factor=dilation_factor)
            except Exception:
                values = (0.025, 0.010, 0.012)
            values_by_class.setdefault(sample.cls_name, []).append(values)

        all_values = [value for values in values_by_class.values() for value in values]
        fallback = _energy_profile_from_values(
            "fallback",
            all_values or [(0.025, 0.010, 0.012)],
            latent_energy_scale=latent_energy_scale,
            latent_energy_floor=latent_energy_floor,
            latent_energy_ceiling=latent_energy_ceiling,
        )
        profiles = {
            cls_name: _energy_profile_from_values(
                cls_name,
                values,
                latent_energy_scale=latent_energy_scale,
                latent_energy_floor=latent_energy_floor,
                latent_energy_ceiling=latent_energy_ceiling,
            )
            for cls_name, values in values_by_class.items()
        }
        return cls(profiles, fallback)

    def profile(self, cls_name: str) -> DefectEnergyProfile:
        return self.profiles.get(cls_name, self.fallback)

    def target_energy(self, cls_name: str, *, seed: int, jitter: float) -> float:
        profile = self.profile(cls_name)
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        span = max(0.0, float(jitter))
        factor = float(rng.uniform(1.0 - span, 1.0 + span)) if span > 0 else 1.0
        low = max(1e-6, profile.latent_target_energy * (1.0 - max(0.25, span)))
        high = max(low, profile.latent_target_energy * (1.0 + max(0.25, span)))
        return float(np.clip(profile.latent_target_energy * factor, low, high))

    def to_manifest(self) -> dict[str, object]:
        return {
            "fallback": asdict(self.fallback),
            "profiles": {cls_name: asdict(profile) for cls_name, profile in sorted(self.profiles.items())},
        }


@dataclass(frozen=True)
class DefectSpectrumObservation:
    cls_name: str
    low_freq_ratio: float
    mid_freq_ratio: float
    high_freq_ratio: float
    orientation_0: float
    orientation_45: float
    orientation_90: float
    orientation_135: float
    polarity: float
    boundary_roughness: float
    component_density: float
    elongation: float
    area_fraction: float
    contrast_ratio: float
    failure_weight: float


@dataclass(frozen=True)
class DefectSpectrumProfile:
    cls_name: str
    sample_count: int
    low_freq_ratio: float
    mid_freq_ratio: float
    high_freq_ratio: float
    orientation_0: float
    orientation_45: float
    orientation_90: float
    orientation_135: float
    polarity: float
    boundary_roughness: float
    component_density: float
    elongation: float
    area_fraction: float
    contrast_ratio: float
    failure_weight: float


class DefectSpectrumPrior:
    """Class-wise defect structure spectra and weak P_fail proxies."""

    def __init__(self, profiles: dict[str, DefectSpectrumProfile], fallback: DefectSpectrumProfile):
        self.profiles = dict(profiles)
        self.fallback = fallback

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[DefectSample],
        *,
        resolution: int,
        dilation_factor: float,
    ) -> "DefectSpectrumPrior":
        values_by_class: dict[str, list[DefectSpectrumObservation]] = {}
        for sample in samples:
            try:
                obs = _estimate_sample_spectrum(sample, resolution=resolution, dilation_factor=dilation_factor)
            except Exception:
                obs = _fallback_spectrum_observation(sample.cls_name)
            values_by_class.setdefault(sample.cls_name, []).append(obs)

        all_values = [value for values in values_by_class.values() for value in values]
        fallback = _spectrum_profile_from_observations("fallback", all_values or [_fallback_spectrum_observation("fallback")])
        profiles = {
            cls_name: _spectrum_profile_from_observations(cls_name, values)
            for cls_name, values in values_by_class.items()
        }
        return cls(profiles, fallback)

    def profile(self, cls_name: str) -> DefectSpectrumProfile:
        return self.profiles.get(cls_name, self.fallback)

    def target_spectrum(
        self,
        cls_name: str,
        *,
        seed: int,
        jitter: float,
        utility_guidance: float,
        coverage_counts: Optional[dict[int, int]] = None,
    ) -> dict[str, float | int]:
        profile = self.profile(cls_name)
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        span = max(0.0, float(jitter))
        utility = float(np.clip(profile.failure_weight, 0.0, 1.0))
        utility_gain = float(np.clip(utility_guidance, 0.0, 1.0)) * utility

        freq = np.asarray(
            [profile.low_freq_ratio, profile.mid_freq_ratio, profile.high_freq_ratio],
            dtype=np.float32,
        )
        freq *= rng.lognormal(mean=0.0, sigma=0.35 * span, size=3).astype(np.float32)
        freq *= np.asarray([1.0 - 0.15 * utility_gain, 1.0 + 0.05 * utility_gain, 1.0 + 0.28 * utility_gain])
        freq = _normalize_vector(freq, fallback=(0.25, 0.35, 0.40))

        orient = np.asarray(
            [profile.orientation_0, profile.orientation_45, profile.orientation_90, profile.orientation_135],
            dtype=np.float32,
        )
        orient *= rng.lognormal(mean=0.0, sigma=0.45 * span, size=4).astype(np.float32)
        if coverage_counts:
            least_seen = min(range(4), key=lambda idx: int(coverage_counts.get(idx, 0)))
            orient[least_seen] += 0.18 + 0.32 * utility_gain
        orient = _normalize_vector(orient, fallback=(0.25, 0.25, 0.25, 0.25))
        orientation_bin = int(np.argmax(orient))

        polarity = float(np.clip(profile.polarity + rng.normal(0.0, 0.35 * span), -1.0, 1.0))
        boundary = float(np.clip(profile.boundary_roughness * rng.uniform(1.0 - span, 1.0 + span), 0.0, 3.0))
        components = float(np.clip(profile.component_density * rng.uniform(1.0 - span, 1.0 + span), 0.0, 4.0))
        elongation = float(np.clip(profile.elongation * rng.uniform(1.0 - 0.6 * span, 1.0 + 0.8 * span), 1.0, 12.0))
        area_fraction = float(np.clip(profile.area_fraction * rng.uniform(1.0 - span, 1.0 + span), 0.002, 0.95))
        contrast_ratio = float(np.clip(profile.contrast_ratio * rng.uniform(1.0 - span, 1.0 + span), 0.05, 20.0))
        energy_multiplier = float(np.clip(1.0 + 0.22 * utility_gain, 0.85, 1.35))

        return {
            "low_freq_ratio": float(freq[0]),
            "mid_freq_ratio": float(freq[1]),
            "high_freq_ratio": float(freq[2]),
            "orientation_0": float(orient[0]),
            "orientation_45": float(orient[1]),
            "orientation_90": float(orient[2]),
            "orientation_135": float(orient[3]),
            "orientation_bin": orientation_bin,
            "polarity": polarity,
            "boundary_roughness": boundary,
            "component_density": components,
            "elongation": elongation,
            "area_fraction": area_fraction,
            "contrast_ratio": contrast_ratio,
            "failure_weight": utility,
            "utility_score": float(np.clip(0.45 + 0.55 * utility_gain, 0.0, 1.0)),
            "energy_multiplier": energy_multiplier,
        }

    def to_manifest(self) -> dict[str, object]:
        return {
            "fallback": asdict(self.fallback),
            "profiles": {cls_name: asdict(profile) for cls_name, profile in sorted(self.profiles.items())},
            "p_fail_proxy": {
                "description": "failure_weight initializes P_fail when detector miss/low-confidence logs are unavailable",
                "signals": ["small area", "weak contrast", "elongation", "boundary roughness", "background clutter"],
            },
        }


def _sample_target_key(sample: DefectSample) -> str:
    return str(getattr(sample, "target_key", sample.image_path.stem))


def _float_mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _normalize_vector(values: np.ndarray, *, fallback: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 1e-8:
        values = np.asarray(list(fallback), dtype=np.float32)
        total = float(values.sum())
    return values / max(total, 1e-8)


def _fallback_spectrum_observation(cls_name: str) -> DefectSpectrumObservation:
    return DefectSpectrumObservation(
        cls_name=cls_name,
        low_freq_ratio=0.25,
        mid_freq_ratio=0.35,
        high_freq_ratio=0.40,
        orientation_0=0.25,
        orientation_45=0.25,
        orientation_90=0.25,
        orientation_135=0.25,
        polarity=0.0,
        boundary_roughness=0.65,
        component_density=0.75,
        elongation=1.8,
        area_fraction=0.18,
        contrast_ratio=1.5,
        failure_weight=0.45,
    )


def _estimate_sample_spectrum(
    sample: DefectSample,
    *,
    resolution: int,
    dilation_factor: float,
) -> DefectSpectrumObservation:
    original = Image.open(sample.image_path).convert("RGB")
    expanded = expand_bbox(sample.bbox, original.size, dilation_factor)
    expanded_crop = original.crop(expanded)
    local_bbox = bbox_inside_crop(sample.bbox, expanded)
    working_crop = expanded_crop.resize((resolution, resolution), Image.Resampling.LANCZOS)
    scaled_bbox = scale_bbox_to_size(local_bbox, expanded_crop.size, working_crop.size)

    gray = np.asarray(working_crop.convert("L"), dtype=np.float32) / 255.0
    x0, y0, x1, y1 = [int(v) for v in scaled_bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    defect = np.zeros_like(gray, dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return _fallback_spectrum_observation(sample.cls_name)
    defect[y0:y1, x0:x1] = True

    blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.1)
    blur3 = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    blur7 = cv2.GaussianBlur(gray, (0, 0), sigmaX=7.0)
    signed = gray - blur3
    low_band = blur7 - float(np.mean(blur7[defect]))
    mid_band = blur1 - blur7
    high_band = gray - blur1
    abs_residual = np.abs(signed)

    inside = abs_residual[defect]
    threshold = float(np.percentile(inside, 58)) if inside.size else 0.0
    core = np.zeros_like(defect, dtype=bool)
    core[defect] = abs_residual[defect] >= threshold
    if int(core.sum()) < max(4, int(0.015 * int(defect.sum()))):
        core = defect

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(defect.astype(np.uint8), kernel, iterations=1).astype(bool)
    background = np.logical_not(dilated)
    low = _rms(low_band[core])
    mid = _rms(mid_band[core])
    high = _rms(high_band[core])
    freq = _normalize_vector(np.asarray([low, mid, high], dtype=np.float32), fallback=(0.25, 0.35, 0.40))

    gx = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(np.square(gx) + np.square(gy))
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    centers = np.asarray([0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0], dtype=np.float32)
    hist = []
    for center in centers:
        diff = np.minimum(np.abs(angle - center), np.pi - np.abs(angle - center))
        weight = np.clip(1.0 - diff / (np.pi / 4.0), 0.0, 1.0)
        hist.append(float((weight[core] * mag[core]).sum()))
    orient = _normalize_vector(np.asarray(hist, dtype=np.float32), fallback=(0.25, 0.25, 0.25, 0.25))

    signed_core = signed[core]
    polarity = float(np.clip(float(np.mean(signed_core)) / max(1e-6, float(np.mean(np.abs(signed_core)))), -1.0, 1.0))
    contours, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(max(1, int(core.sum())))
    perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    boundary_roughness = float(np.clip(perimeter / max(1e-6, 2.0 * math.sqrt(math.pi * area)) - 1.0, 0.0, 3.0))
    n_labels, _ = cv2.connectedComponents(core.astype(np.uint8), connectivity=8)
    bbox_area = float(max(1, int(defect.sum())))
    component_density = float(np.clip(max(0, int(n_labels) - 1) / max(1.0, bbox_area / 2048.0), 0.0, 4.0))
    width = max(1.0, float(x1 - x0))
    height = max(1.0, float(y1 - y0))
    elongation = float(np.clip(max(width / height, height / width), 1.0, 12.0))
    area_fraction = float(np.clip(area / bbox_area, 0.002, 0.95))
    core_energy = _rms(signed[core])
    background_energy = _rms(signed[background]) if background.any() else 0.01
    contrast_ratio = float(np.clip(core_energy / max(1e-4, background_energy), 0.05, 20.0))

    smallness = float(np.clip(1.0 - math.sqrt(max(0.0, area_fraction)), 0.0, 1.0))
    weak_contrast = float(np.clip(math.exp(-contrast_ratio / 1.8), 0.0, 1.0))
    elongation_hard = float(np.clip((elongation - 1.0) / 6.0, 0.0, 1.0))
    rough_hard = float(np.clip(boundary_roughness / 1.6, 0.0, 1.0))
    clutter_hard = float(np.clip(background_energy / max(core_energy + background_energy, 1e-6), 0.0, 1.0))
    failure_weight = float(
        np.clip(
            0.30 * smallness
            + 0.28 * weak_contrast
            + 0.18 * elongation_hard
            + 0.14 * rough_hard
            + 0.10 * clutter_hard,
            0.0,
            1.0,
        )
    )

    return DefectSpectrumObservation(
        cls_name=sample.cls_name,
        low_freq_ratio=float(freq[0]),
        mid_freq_ratio=float(freq[1]),
        high_freq_ratio=float(freq[2]),
        orientation_0=float(orient[0]),
        orientation_45=float(orient[1]),
        orientation_90=float(orient[2]),
        orientation_135=float(orient[3]),
        polarity=polarity,
        boundary_roughness=boundary_roughness,
        component_density=component_density,
        elongation=elongation,
        area_fraction=area_fraction,
        contrast_ratio=contrast_ratio,
        failure_weight=failure_weight,
    )


def _spectrum_profile_from_observations(
    cls_name: str,
    observations: Sequence[DefectSpectrumObservation],
) -> DefectSpectrumProfile:
    if not observations:
        observations = [_fallback_spectrum_observation(cls_name)]

    def median(field: str) -> float:
        return float(np.median(np.asarray([getattr(obs, field) for obs in observations], dtype=np.float32)))

    freq = _normalize_vector(
        np.asarray(
            [median("low_freq_ratio"), median("mid_freq_ratio"), median("high_freq_ratio")],
            dtype=np.float32,
        ),
        fallback=(0.25, 0.35, 0.40),
    )
    orient = _normalize_vector(
        np.asarray(
            [median("orientation_0"), median("orientation_45"), median("orientation_90"), median("orientation_135")],
            dtype=np.float32,
        ),
        fallback=(0.25, 0.25, 0.25, 0.25),
    )
    return DefectSpectrumProfile(
        cls_name=cls_name,
        sample_count=len(observations),
        low_freq_ratio=float(freq[0]),
        mid_freq_ratio=float(freq[1]),
        high_freq_ratio=float(freq[2]),
        orientation_0=float(orient[0]),
        orientation_45=float(orient[1]),
        orientation_90=float(orient[2]),
        orientation_135=float(orient[3]),
        polarity=median("polarity"),
        boundary_roughness=median("boundary_roughness"),
        component_density=median("component_density"),
        elongation=median("elongation"),
        area_fraction=median("area_fraction"),
        contrast_ratio=median("contrast_ratio"),
        failure_weight=median("failure_weight"),
    )


def _estimate_sample_energy(
    sample: DefectSample,
    *,
    resolution: int,
    dilation_factor: float,
) -> tuple[float, float, float]:
    original = Image.open(sample.image_path).convert("RGB")
    expanded = expand_bbox(sample.bbox, original.size, dilation_factor)
    expanded_crop = original.crop(expanded)
    local_bbox = bbox_inside_crop(sample.bbox, expanded)
    working_crop = expanded_crop.resize((resolution, resolution), Image.Resampling.LANCZOS)
    scaled_bbox = scale_bbox_to_size(local_bbox, expanded_crop.size, working_crop.size)

    gray = np.asarray(working_crop.convert("L"), dtype=np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    residual = np.abs(gray - smooth)
    x0, y0, x1, y1 = [int(v) for v in scaled_bbox]
    defect = np.zeros_like(residual, dtype=bool)
    defect[y0:y1, x0:x1] = True
    if not defect.any():
        return 0.025, 0.010, 0.012

    inside = residual[defect]
    threshold = float(np.percentile(inside, 65)) if inside.size else 0.0
    core = np.zeros_like(defect, dtype=bool)
    core[defect] = residual[defect] >= threshold
    if int(core.sum()) < max(4, int(0.01 * int(defect.sum()))):
        core = defect

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(defect.astype(np.uint8), kernel, iterations=1).astype(bool)
    shell = np.logical_and(dilated, np.logical_not(defect))
    background = np.logical_not(dilated)
    core_energy = _rms(residual[core])
    background_energy = _rms(residual[background]) if background.any() else 0.0
    boundary_energy = _rms(residual[shell]) if shell.any() else background_energy
    return core_energy, background_energy, boundary_energy


def _energy_profile_from_values(
    cls_name: str,
    values: Sequence[tuple[float, float, float]],
    *,
    latent_energy_scale: float,
    latent_energy_floor: float,
    latent_energy_ceiling: float,
) -> DefectEnergyProfile:
    core = np.asarray([v[0] for v in values], dtype=np.float32)
    background = np.asarray([v[1] for v in values], dtype=np.float32)
    boundary = np.asarray([v[2] for v in values], dtype=np.float32)
    if core.size == 0:
        core = np.asarray([0.025], dtype=np.float32)
        background = np.asarray([0.010], dtype=np.float32)
        boundary = np.asarray([0.012], dtype=np.float32)
    q25, q50, q75 = [float(np.percentile(core, q)) for q in (25, 50, 75)]
    bg_q75 = float(np.percentile(background, 75))
    boundary_q50 = float(np.percentile(boundary, 50))
    robust_core = max(q50, 0.65 * q75, 0.015)
    latent_target = float(np.clip(robust_core * float(latent_energy_scale), latent_energy_floor, latent_energy_ceiling))
    return DefectEnergyProfile(
        cls_name=cls_name,
        sample_count=int(core.size),
        core_energy_q25=q25,
        core_energy_q50=q50,
        core_energy_q75=q75,
        background_energy_q75=bg_q75,
        boundary_energy_q50=boundary_q50,
        latent_target_energy=latent_target,
    )


def _build_lgi_masks(shape_mask: Image.Image, *, shell_radius: int = 9) -> LGIMaskSet:
    mask = np.asarray(shape_mask.convert("L"), dtype=np.float32) / 255.0
    mask = np.clip(mask, 0.0, 1.0)
    core = np.clip(mask, 0.0, 1.0)
    binary = core > 0.08
    if binary.any():
        kernel_size = max(3, int(shell_radius) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        shell = np.logical_and(dilated, np.logical_not(binary)).astype(np.float32)
        shell = cv2.GaussianBlur(shell, (0, 0), sigmaX=max(0.7, shell_radius / 5.0))
    else:
        shell = np.zeros_like(core, dtype=np.float32)
    shell = np.clip(shell * (1.0 - core), 0.0, 1.0)
    edit = np.clip(np.maximum(core, shell), 0.0, 1.0)
    out = np.clip(1.0 - edit, 0.0, 1.0)
    denom = float(max(1, core.size))
    stats = {
        "core_coverage": float(np.count_nonzero(core > 0.08) / denom),
        "shell_coverage": float(np.count_nonzero(shell > 0.05) / denom),
        "edit_coverage": float(np.count_nonzero(edit > 0.03) / denom),
        "out_coverage": float(np.count_nonzero(out > 0.5) / denom),
    }
    return LGIMaskSet(
        core=_float_mask_image(core),
        shell=_float_mask_image(shell),
        out=_float_mask_image(out),
        edit=_float_mask_image(edit),
        stats=stats,
    )


def _srt_family_from_class(cls_name: str) -> str:
    cls = cls_name.lower()
    if "scratch" in cls:
        return "scratch"
    if "crazing" in cls or "crack" in cls:
        return "crack"
    if "pit" in cls:
        return "pitted"
    if "inclusion" in cls:
        return "inclusion"
    if "patch" in cls:
        return "patch"
    if "scale" in cls or "rolled" in cls:
        return "scale"
    return "blob"


def _clip_bbox_to_size(
    bbox: Sequence[float | int],
    size: tuple[int, int],
    *,
    min_size: int = 2,
) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    x0 = max(0, min(max(0, width - min_size), x0))
    y0 = max(0, min(max(0, height - min_size), y0))
    x1 = max(x0 + min_size, min(width, x1))
    y1 = max(y0 + min_size, min(height, y1))
    return int(x0), int(y0), int(x1), int(y1)


def _fit_bbox_inside(
    bbox: Sequence[float | int],
    bounds: Sequence[float | int],
    size: tuple[int, int],
    *,
    min_size: int = 2,
) -> tuple[int, int, int, int]:
    width, height = size
    bx0, by0, bx1, by1 = _clip_bbox_to_size(bounds, size, min_size=min_size)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw = min(max(float(min_size), x1 - x0), max(float(min_size), float(bx1 - bx0)))
    bh = min(max(float(min_size), y1 - y0), max(float(min_size), float(by1 - by0)))
    cx = float(np.clip((x0 + x1) * 0.5, bx0 + bw * 0.5, bx1 - bw * 0.5))
    cy = float(np.clip((y0 + y1) * 0.5, by0 + bh * 0.5, by1 - bh * 0.5))
    return _clip_bbox_to_size(
        (cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5),
        (width, height),
        min_size=min_size,
    )


def _binary_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _robust_binary_bbox(mask: np.ndarray, *, low_q: float = 2.5, high_q: float = 97.5) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(np.floor(np.percentile(xs, low_q)))
    x1 = int(np.ceil(np.percentile(xs, high_q))) + 1
    y0 = int(np.floor(np.percentile(ys, low_q)))
    y1 = int(np.ceil(np.percentile(ys, high_q))) + 1
    return x0, y0, x1, y1


def _bbox_iou(a: Sequence[int | float], b: Sequence[int | float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return float(inter / max(1e-6, area_a + area_b - inter))


def _bbox_area_fraction(
    bbox: Sequence[int | float],
    bounds: Sequence[int | float],
) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bx0, by0, bx1, by1 = [float(v) for v in bounds]
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    bound_area = max(1.0, max(0.0, bx1 - bx0) * max(0.0, by1 - by0))
    return float(np.clip(area / bound_area, 0.0, 1.0))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    return float(inter / max(1, union))


def _constrain_binary_to_bbox(
    binary: np.ndarray,
    bbox: Sequence[int | float],
) -> np.ndarray:
    height, width = binary.shape[:2]
    x0, y0, x1, y1 = _clip_bbox_to_size(bbox, (width, height), min_size=1)
    out = np.zeros_like(binary, dtype=bool)
    out[y0:y1, x0:x1] = binary[y0:y1, x0:x1].astype(bool)
    return out


def _orientation_angle_from_spectrum(target_spectrum: dict[str, float | int]) -> float:
    hist = np.asarray(
        [
            float(target_spectrum.get("orientation_0", 0.25)),
            float(target_spectrum.get("orientation_45", 0.25)),
            float(target_spectrum.get("orientation_90", 0.25)),
            float(target_spectrum.get("orientation_135", 0.25)),
        ],
        dtype=np.float32,
    )
    angles = np.asarray([0.0, 45.0, 90.0, 135.0], dtype=np.float32)
    if float(hist.sum()) <= 1e-6:
        return 0.0
    return float(angles[int(np.argmax(hist))])


def _skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    binary = (mask.astype(np.uint8) > 0).astype(np.uint8)
    skeleton = np.zeros_like(binary, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    max_iter = int(max(mask.shape) * 2)
    for _ in range(max_iter):
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(binary, opened)
        eroded = cv2.erode(binary, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded
        if int(cv2.countNonZero(binary)) == 0:
            break
    return skeleton.astype(bool)


def _structure_metrics(mask: np.ndarray) -> dict[str, float | int]:
    binary = mask.astype(bool)
    area = int(binary.sum())
    total = max(1, int(binary.size))
    if area <= 0:
        return {
            "area_fraction": 0.0,
            "component_count": 0,
            "skeleton_length": 0.0,
            "mean_width": 0.0,
            "boundary_roughness": 0.0,
            "orientation_deg": 0.0,
        }
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    min_area = max(3, int(total * 0.0015))
    component_count = sum(1 for idx in range(1, num_labels) if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area)
    contours, _hier = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    roughness = float(np.clip(perimeter / max(1e-6, 2.0 * math.sqrt(math.pi * max(1.0, float(area)))) - 1.0, 0.0, 5.0))
    skeleton = _skeletonize_binary(binary)
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    skeleton_values = dist[skeleton]
    mean_width = float(2.0 * float(np.mean(skeleton_values))) if skeleton_values.size else float(2.0 * math.sqrt(area / math.pi))
    orientation_deg = 0.0
    pts = np.column_stack(np.where(binary))
    if pts.shape[0] >= 3:
        pts = pts.astype(np.float32)
        pts[:, 0] -= float(np.mean(pts[:, 0]))
        pts[:, 1] -= float(np.mean(pts[:, 1]))
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        vec = eigvecs[:, int(np.argmax(eigvals))]
        orientation_deg = float(np.degrees(np.arctan2(vec[0], vec[1])))
    return {
        "area_fraction": float(area / total),
        "component_count": int(max(1, component_count)),
        "skeleton_length": float(skeleton.sum()),
        "mean_width": float(mean_width),
        "boundary_roughness": float(roughness),
        "orientation_deg": float(orientation_deg),
    }


def _angle_periodic_distance_deg(a: float, b: float) -> float:
    return float(abs(((float(a) - float(b) + 90.0) % 180.0) - 90.0))


def _angle_match_score(a: float, b: float, *, scale: float = 42.0) -> float:
    return float(math.exp(-_angle_periodic_distance_deg(a, b) / max(1e-6, float(scale))))


def _bbox_elongation_from_binary(mask: np.ndarray, fallback_bbox: Sequence[int | float]) -> float:
    bbox = _robust_binary_bbox(mask, low_q=1.0, high_q=99.0) or _binary_bbox(mask)
    if bbox is None:
        bbox = _clip_bbox_to_size(fallback_bbox, (mask.shape[1], mask.shape[0]), min_size=2)
    x0, y0, x1, y1 = bbox
    width = max(1.0, float(x1 - x0))
    height = max(1.0, float(y1 - y0))
    return float(np.clip(max(width / height, height / width), 1.0, 12.0))


def _log_match_score(value: float, target: float, *, scale: float) -> float:
    value = max(1e-6, float(value))
    target = max(1e-6, float(target))
    return float(math.exp(-abs(math.log(value / target)) / max(1e-6, float(scale))))


def _hdsi_s2c_structure_alignment(
    binary: np.ndarray,
    label_bbox: Sequence[int | float],
    *,
    target_spectrum: dict[str, float | int],
    target_fraction: float,
) -> dict[str, float]:
    """Measure whether phi' embodies the HDSI target spectrum before image decoding."""

    constrained = _constrain_binary_to_bbox(binary, label_bbox)
    metrics = _structure_metrics(constrained)
    local_fraction = max(1e-6, _mask_local_area_fraction(constrained, label_bbox))
    target_angle = _orientation_angle_from_spectrum(target_spectrum)
    orientation_score = _angle_match_score(float(metrics.get("orientation_deg", 0.0)), target_angle)
    area_score = _log_match_score(local_fraction, max(1e-6, float(target_fraction)), scale=0.82)

    roughness = float(metrics.get("boundary_roughness", 0.0))
    target_roughness = float(np.clip(float(target_spectrum.get("boundary_roughness", 0.65)), 0.0, 3.0))
    roughness_score = float(math.exp(-abs(roughness - target_roughness) / 1.10))

    component_count = float(metrics.get("component_count", 0.0))
    target_density = float(np.clip(float(target_spectrum.get("component_density", 0.75)), 0.0, 4.0))
    target_components = 1.0 + 1.75 * target_density
    density_score = float(math.exp(-abs(component_count - target_components) / 3.25))

    elongation = _bbox_elongation_from_binary(constrained, label_bbox)
    target_elongation = float(np.clip(float(target_spectrum.get("elongation", 1.8)), 1.0, 12.0))
    elongation_score = _log_match_score(elongation, target_elongation, scale=0.96)

    score = float(
        np.clip(
            0.24 * area_score
            + 0.24 * orientation_score
            + 0.18 * roughness_score
            + 0.16 * density_score
            + 0.18 * elongation_score,
            0.0,
            1.0,
        )
    )
    return {
        "hdsi_s2c_structure_score": score,
        "hdsi_s2c_area_score": float(area_score),
        "hdsi_s2c_orientation_score": float(orientation_score),
        "hdsi_s2c_roughness_score": float(roughness_score),
        "hdsi_s2c_density_score": float(density_score),
        "hdsi_s2c_elongation_score": float(elongation_score),
        "hdsi_s2c_structure_local_fraction": float(local_fraction),
        "hdsi_s2c_target_local_fraction": float(target_fraction),
    }


def _mask_local_area_fraction(mask: np.ndarray, bbox: Sequence[int | float]) -> float:
    height, width = mask.shape[:2]
    x0, y0, x1, y1 = _clip_bbox_to_size(bbox, (width, height), min_size=1)
    region = mask[y0:y1, x0:x1].astype(bool)
    return float(region.mean()) if region.size else 0.0


def _srt_target_local_fraction(
    family: str,
    target_spectrum: dict[str, float | int],
    *,
    source_fraction: float,
) -> float:
    requested = float(np.clip(float(target_spectrum.get("area_fraction", source_fraction)), 0.004, 0.92))
    if family in {"scratch"}:
        low, high = 0.018, 0.16
    elif family in {"crack"}:
        low, high = 0.030, 0.24
    elif family in {"inclusion"}:
        low, high = 0.030, 0.22
    elif family in {"pitted"}:
        low, high = 0.035, 0.26
    elif family in {"patch", "scale"}:
        low, high = 0.075, 0.40
    else:
        low, high = 0.045, 0.32
    blended = 0.55 * requested + 0.45 * float(np.clip(source_fraction, low, high))
    return float(np.clip(blended, low, high))


def _cap_structure_local_area(
    binary: np.ndarray,
    bbox: Sequence[int | float],
    *,
    target_fraction: float,
    family: str,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = binary.shape[:2]
    x0, y0, x1, y1 = _clip_bbox_to_size(bbox, (width, height), min_size=2)
    crop = binary[y0:y1, x0:x1].astype(bool)
    if crop.size == 0 or not crop.any():
        return binary.astype(bool)
    current = float(crop.mean())
    max_fraction = float(np.clip(target_fraction * (1.22 if family in {"crack", "scratch"} else 1.35), 0.006, 0.58))
    min_fraction = float(np.clip(target_fraction * 0.42, 0.003, max_fraction))
    out = binary.astype(bool).copy()
    if current > max_fraction:
        dist = cv2.distanceTransform(crop.astype(np.uint8), cv2.DIST_L2, 3)
        if family in {"crack", "scratch"}:
            skeleton = _skeletonize_binary(crop)
            dist = dist + skeleton.astype(np.float32) * max(1.0, float(dist.max()))
        noise = rng.normal(0.0, 0.12, crop.shape).astype(np.float32)
        score = dist + noise
        keep_count = max(3, int(round(max_fraction * crop.size)))
        values = score[crop]
        if values.size > keep_count:
            threshold = float(np.partition(values, values.size - keep_count)[values.size - keep_count])
            crop = np.logical_and(crop, score >= threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if family in {"crack", "scratch"}:
            crop = cv2.dilate(crop.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            crop = cv2.morphologyEx(crop.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    elif current < min_fraction:
        iterations = 1 if current > min_fraction * 0.45 else 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        crop = cv2.dilate(crop.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    out[y0:y1, x0:x1] = crop
    return out


def _soft_structure_mask(binary: np.ndarray, *, roughness: float) -> np.ndarray:
    hard = binary.astype(np.uint8)
    if int(hard.sum()) == 0:
        return hard.astype(np.float32)
    inside = cv2.distanceTransform(hard, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((1 - hard).astype(np.uint8), cv2.DIST_L2, 3)
    sdf = inside - outside
    softness = max(1.4, 3.6 - 0.48 * float(roughness))
    soft = 1.0 / (1.0 + np.exp(-sdf / softness))
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _estimate_source_structure_mask(
    working_crop: Image.Image,
    scaled_bbox: Sequence[int | float],
    fallback_mask: Image.Image,
) -> np.ndarray:
    gray = np.asarray(working_crop.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    x0, y0, x1, y1 = _clip_bbox_to_size(scaled_bbox, (w, h), min_size=2)
    region = np.zeros_like(gray, dtype=bool)
    region[y0:y1, x0:x1] = True
    blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    blur4 = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.2)
    residual = np.abs(gray - blur4) + 0.35 * np.abs(blur1 - blur4)
    inside = residual[region]
    if inside.size == 0:
        return np.asarray(fallback_mask.convert("L"), dtype=np.float32) > 32
    threshold = float(np.percentile(inside, 72))
    binary = np.zeros_like(region, dtype=np.uint8)
    binary[region] = (residual[region] >= threshold).astype(np.uint8)
    if int(binary.sum()) < max(6, int(region.sum() * 0.012)):
        threshold = float(np.percentile(inside, 48))
        binary[region] = (residual[region] >= threshold).astype(np.uint8)
    coverage = float(binary[region].mean()) if region.any() else 0.0
    if coverage > 0.24:
        threshold = float(np.percentile(inside, 84))
        binary = np.zeros_like(region, dtype=np.uint8)
        binary[region] = (residual[region] >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if int(binary.sum()) < max(5, int(region.sum() * 0.006)):
        binary = (np.asarray(fallback_mask.convert("L"), dtype=np.float32) > 32).astype(np.uint8)
    binary = np.logical_and(binary.astype(bool), region)
    if not binary.any():
        fallback = np.zeros_like(binary, dtype=np.uint8)
        cv2.rectangle(fallback, (x0, y0), (x1 - 1, y1 - 1), 1, thickness=-1)
        binary = fallback.astype(bool)
    return binary.astype(bool)


def _warp_binary_to_bbox(
    source_mask: np.ndarray,
    source_bbox: Sequence[int | float],
    target_bbox: Sequence[int | float],
    *,
    angle_delta: float,
    output_size: tuple[int, int],
) -> np.ndarray:
    width, height = output_size
    sx0, sy0, sx1, sy1 = _clip_bbox_to_size(source_bbox, output_size, min_size=2)
    tx0, ty0, tx1, ty1 = _clip_bbox_to_size(target_bbox, output_size, min_size=2)
    crop = (source_mask[sy0:sy1, sx0:sx1].astype(np.uint8) * 255)
    if crop.size == 0:
        return np.zeros((height, width), dtype=bool)
    crop_img = Image.fromarray(crop, mode="L")
    if abs(float(angle_delta)) > 1.0:
        crop_img = crop_img.rotate(float(angle_delta), resample=Image.Resampling.BILINEAR, expand=False)
    tw, th = max(2, tx1 - tx0), max(2, ty1 - ty0)
    crop_img = crop_img.resize((tw, th), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (width, height), 0)
    canvas.paste(crop_img, (tx0, ty0))
    return (np.asarray(canvas, dtype=np.uint8) > 32)


def _add_srt_structural_events(
    binary: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    family: str,
    target_spectrum: dict[str, float | int],
    rng: np.random.Generator,
    config: DELGIConfig,
) -> np.ndarray:
    h, w = binary.shape
    x0, y0, x1, y1 = bbox
    bw, bh = max(2, x1 - x0), max(2, y1 - y0)
    canvas = (binary.astype(np.uint8) * 255).copy()
    strength = float(np.clip(config.srt_component_jitter, 0.0, 1.0))
    component_density = float(np.clip(float(target_spectrum.get("component_density", 1.0)), 0.0, 4.0))
    polarity = float(np.clip(float(target_spectrum.get("polarity", 0.0)), -1.0, 1.0))
    event_count = int(np.clip(round(1 + component_density + 3.0 * strength), 1, 8))
    angle = math.radians(_orientation_angle_from_spectrum(target_spectrum))

    if family in {"scratch", "crack"}:
        for _ in range(max(1, min(3, event_count))):
            cx = float(rng.uniform(x0 + 0.18 * bw, x1 - 0.18 * bw))
            cy = float(rng.uniform(y0 + 0.18 * bh, y1 - 0.18 * bh))
            length = float(rng.uniform(0.55, 1.10) * math.hypot(bw, bh))
            thickness = int(np.clip(round(rng.uniform(1.0, 2.6 + 3.0 * strength)), 1, max(2, int(min(bw, bh) * 0.12))))
            steps = int(rng.integers(4, 8))
            points = []
            local_angle = angle + float(rng.normal(0.0, 0.22 + 0.18 * strength))
            px = cx - math.cos(local_angle) * length * 0.45
            py = cy - math.sin(local_angle) * length * 0.45
            for _idx in range(steps + 1):
                points.append([int(np.clip(px, x0, x1 - 1)), int(np.clip(py, y0, y1 - 1))])
                local_angle += float(rng.normal(0.0, 0.08 + 0.12 * strength))
                px += math.cos(local_angle) * length / max(1, steps)
                py += math.sin(local_angle) * length / max(1, steps)
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
    elif family in {"pitted", "inclusion"}:
        for _ in range(event_count):
            cx = int(rng.integers(x0, max(x0 + 1, x1)))
            cy = int(rng.integers(y0, max(y0 + 1, y1)))
            rx = int(np.clip(round(rng.uniform(0.018, 0.090) * bw), 1, max(2, bw // 3)))
            ry = int(np.clip(round(rng.uniform(0.018, 0.120) * bh), 1, max(2, bh // 3)))
            rot = float(rng.uniform(-35.0, 35.0))
            cv2.ellipse(canvas, (cx, cy), (rx, ry), rot, 0, 360, 255, thickness=-1)
    else:
        for _ in range(max(2, event_count)):
            cx = int(rng.integers(x0, max(x0 + 1, x1)))
            cy = int(rng.integers(y0, max(y0 + 1, y1)))
            rx = int(np.clip(round(rng.uniform(0.055, 0.22) * bw), 2, max(3, bw // 2)))
            ry = int(np.clip(round(rng.uniform(0.035, 0.18) * bh), 2, max(3, bh // 2)))
            cv2.ellipse(canvas, (cx, cy), (rx, ry), float(rng.uniform(-45, 45)), 0, 360, 255, thickness=-1)

    binary = canvas > 32
    if float(polarity) < -0.25 and family in {"inclusion", "pitted", "blob"}:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    return binary


def _draw_intra_box_counterfactual_structure(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    family: str,
    target_spectrum: dict[str, float | int],
    rng: np.random.Generator,
    config: DELGIConfig,
) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = bbox
    bw, bh = max(2, x1 - x0), max(2, y1 - y0)
    canvas = np.zeros((height, width), dtype=np.uint8)
    strength = float(np.clip(config.srt_component_jitter, 0.0, 1.0))
    component_density = float(np.clip(float(target_spectrum.get("component_density", 1.0)), 0.0, 4.0))
    angle = math.radians(_orientation_angle_from_spectrum(target_spectrum) + float(rng.normal(0.0, 18.0 * strength)))

    if family in {"scratch", "crack"}:
        line_count = int(np.clip(round(1.0 + 0.65 * component_density + 2.4 * strength), 1, 5 if family == "crack" else 3))
        diag = math.hypot(bw, bh)
        for line_idx in range(line_count):
            local_angle = angle + float(rng.normal(0.0, 0.20 + 0.18 * strength))
            direction = np.asarray([math.cos(local_angle), math.sin(local_angle)], dtype=np.float32)
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
            center = np.asarray(
                [
                    rng.uniform(x0 + 0.18 * bw, x1 - 0.18 * bw),
                    rng.uniform(y0 + 0.18 * bh, y1 - 0.18 * bh),
                ],
                dtype=np.float32,
            )
            length = float(rng.uniform(0.64, 1.14) * diag)
            steps = int(rng.integers(7, 14))
            curvature = float(rng.normal(0.0, (0.06 if family == "scratch" else 0.13) * diag * (0.35 + strength)))
            phase = float(rng.uniform(0.0, math.pi))
            points: list[list[int]] = []
            drift = 0.0
            for idx in range(steps + 1):
                t = idx / max(1, steps)
                drift += float(rng.normal(0.0, 0.012 * diag * strength))
                pos = center + direction * ((t - 0.5) * length) + normal * (math.sin(t * math.pi + phase) * curvature + drift)
                points.append([
                    int(np.clip(round(float(pos[0])), x0, x1 - 1)),
                    int(np.clip(round(float(pos[1])), y0, y1 - 1)),
                ])
            thickness = int(np.clip(round(rng.uniform(1.0, 2.8 + 3.5 * strength)), 1, max(2, int(min(bw, bh) * 0.16))))
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
            if family == "crack" and rng.random() < 0.55 + 0.25 * strength and len(points) >= 4:
                anchor = np.asarray(points[int(rng.integers(1, len(points) - 2))], dtype=np.float32)
                branch_angle = local_angle + float(rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 1.05))
                branch_dir = np.asarray([math.cos(branch_angle), math.sin(branch_angle)], dtype=np.float32)
                branch_len = float(rng.uniform(0.16, 0.38) * diag)
                branch_points = []
                for bid in range(4):
                    pos = anchor + branch_dir * (branch_len * bid / 3.0)
                    branch_points.append([
                        int(np.clip(round(float(pos[0])), x0, x1 - 1)),
                        int(np.clip(round(float(pos[1])), y0, y1 - 1)),
                    ])
                cv2.polylines(canvas, [np.asarray(branch_points, dtype=np.int32)], False, 255, thickness=max(1, thickness - 1), lineType=cv2.LINE_AA)
    elif family in {"pitted", "inclusion"}:
        cluster_count = int(np.clip(round(1.0 + component_density + 2.0 * strength), 2, 8))
        dot_count = int(np.clip(round(cluster_count * rng.uniform(2.5, 5.5)), 5, 34))
        centers = [
            (float(rng.uniform(x0 + 0.10 * bw, x1 - 0.10 * bw)), float(rng.uniform(y0 + 0.10 * bh, y1 - 0.10 * bh)))
            for _ in range(cluster_count)
        ]
        for idx in range(dot_count):
            ccx, ccy = centers[int(rng.integers(0, len(centers)))]
            cx = int(np.clip(round(rng.normal(ccx, 0.12 * bw)), x0, x1 - 1))
            cy = int(np.clip(round(rng.normal(ccy, 0.12 * bh)), y0, y1 - 1))
            if family == "pitted":
                rx = int(np.clip(round(rng.uniform(0.010, 0.045) * bw), 1, max(2, bw // 8)))
                ry = int(np.clip(round(rng.uniform(0.010, 0.055) * bh), 1, max(2, bh // 8)))
            else:
                rx = int(np.clip(round(rng.uniform(0.020, 0.090) * bw), 1, max(3, bw // 4)))
                ry = int(np.clip(round(rng.uniform(0.030, 0.150) * bh), 1, max(3, bh // 3)))
            cv2.ellipse(canvas, (cx, cy), (rx, ry), float(rng.uniform(-50, 50)), 0, 360, 255, thickness=-1)
    else:
        blob_count = int(np.clip(round(1.0 + component_density + 2.6 * strength), 2, 9))
        for _ in range(blob_count):
            cx = float(rng.uniform(x0 + 0.12 * bw, x1 - 0.12 * bw))
            cy = float(rng.uniform(y0 + 0.12 * bh, y1 - 0.12 * bh))
            radius_x = float(rng.uniform(0.08, 0.26) * bw)
            radius_y = float(rng.uniform(0.05, 0.22) * bh)
            vertex_count = int(rng.integers(7, 14))
            angles = np.linspace(0.0, 2.0 * math.pi, vertex_count, endpoint=False) + float(rng.uniform(0.0, 0.45))
            pts = []
            for a in angles:
                scale = float(rng.uniform(0.58, 1.26))
                px = cx + math.cos(float(a)) * radius_x * scale
                py = cy + math.sin(float(a)) * radius_y * scale
                pts.append([int(np.clip(round(px), x0, x1 - 1)), int(np.clip(round(py), y0, y1 - 1))])
            cv2.fillPoly(canvas, [np.asarray(pts, dtype=np.int32)], 255, lineType=cv2.LINE_AA)

    return (canvas > 32)


def _sample_intra_box_counterfactual_structure(
    *,
    source_binary: np.ndarray,
    prior_binary: np.ndarray,
    label_bbox: tuple[int, int, int, int],
    family: str,
    target_spectrum: dict[str, float | int],
    target_fraction: float,
    rng: np.random.Generator,
    config: DELGIConfig,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Sample a class-consistent but source-different structure inside the inherited label bbox."""

    height, width = source_binary.shape[:2]
    source_in_label = _constrain_binary_to_bbox(source_binary, label_bbox)
    prior_in_label = _constrain_binary_to_bbox(prior_binary, label_bbox)
    best: tuple[float, np.ndarray, dict[str, float | str]] | None = None
    preset_key = str(config.method_preset).lower().replace("_", "-")
    novelty_preset = preset_key == "e-srt-hdsi-pd"
    attempts = 14 if novelty_preset else 8
    strength = float(np.clip(config.srt_component_jitter, 0.0, 1.0))
    for attempt in range(attempts):
        candidate = _draw_intra_box_counterfactual_structure(
            (width, height),
            label_bbox,
            family=family,
            target_spectrum=target_spectrum,
            rng=rng,
            config=config,
        )
        if prior_in_label.any():
            prior_keep = (0.05 + 0.10 * strength) if novelty_preset else (0.10 + 0.18 * strength)
            candidate = np.logical_or(candidate, np.logical_and(prior_in_label, rng.random(prior_in_label.shape) < prior_keep))
        candidate = _roughen_structure_boundary(
            candidate,
            rng=rng,
            roughness_strength=float(np.clip(config.srt_boundary_roughness, 0.0, 1.0)),
        )
        candidate = _cap_structure_local_area(
            candidate,
            label_bbox,
            target_fraction=target_fraction,
            family=family,
            rng=rng,
        )
        candidate = _constrain_binary_to_bbox(candidate, label_bbox)
        if int(candidate.sum()) < 3:
            continue
        bbox = _robust_binary_bbox(candidate, low_q=1.0, high_q=99.0) or _binary_bbox(candidate) or label_bbox
        local_fraction = _mask_local_area_fraction(candidate, label_bbox)
        area_score = 1.0 - min(1.0, abs(local_fraction - target_fraction) / max(0.012, target_fraction))
        source_iou = _mask_iou(candidate, source_in_label)
        novelty = 1.0 - source_iou
        coverage = _bbox_area_fraction(bbox, label_bbox)
        component_count = float(_structure_metrics(candidate).get("component_count", 0))
        component_score = float(np.clip(component_count / (2.0 + 4.0 * strength), 0.0, 1.0))
        if novelty_preset:
            coverage_target = 0.42 if family in {"scratch", "crack"} else 0.55
            coverage_novelty = float(np.clip(abs(coverage - coverage_target) / max(0.12, coverage_target), 0.0, 1.0))
            score = (
                0.58 * novelty
                + 0.16 * area_score
                + 0.12 * coverage
                + 0.10 * component_score
                + 0.04 * coverage_novelty
            )
        else:
            score = 0.50 * novelty + 0.22 * area_score + 0.18 * coverage + 0.10 * component_score
        info = {
            "counterfactual_generation": "label_consistent_intra_box",
            "counterfactual_structure_iou": float(source_iou),
            "counterfactual_structure_novelty": float(novelty),
            "active_bbox_label_coverage": float(coverage),
            "counterfactual_area_score": float(area_score),
            "counterfactual_attempts": float(attempt + 1),
        }
        if best is None or score > best[0]:
            best = (score, candidate, info)
    if best is not None:
        return best[1], best[2]

    fallback = prior_in_label if prior_in_label.any() else source_in_label
    fallback = _cap_structure_local_area(
        fallback,
        label_bbox,
        target_fraction=target_fraction,
        family=family,
        rng=rng,
    )
    return _constrain_binary_to_bbox(fallback, label_bbox), {
        "counterfactual_generation": "label_consistent_intra_box_fallback",
        "counterfactual_structure_iou": float(_mask_iou(fallback, source_in_label)),
        "counterfactual_structure_novelty": float(1.0 - _mask_iou(fallback, source_in_label)),
        "active_bbox_label_coverage": float(_bbox_area_fraction(_binary_bbox(fallback) or label_bbox, label_bbox)),
        "counterfactual_area_score": 0.0,
        "counterfactual_attempts": float(attempts),
    }


def _roughen_structure_boundary(
    binary: np.ndarray,
    *,
    rng: np.random.Generator,
    roughness_strength: float,
) -> np.ndarray:
    if not binary.any() or roughness_strength <= 0:
        return binary.astype(bool)
    hard = binary.astype(np.uint8)
    dilated = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    eroded = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    band = np.logical_xor(dilated > 0, eroded > 0)
    noise = rng.normal(0.0, 1.0, binary.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(0.65, 2.1 - 1.25 * roughness_strength))
    add = np.logical_and(band, noise > (0.42 - 0.52 * roughness_strength))
    remove = np.logical_and(binary, np.logical_and(band, noise < (-0.58 + 0.32 * roughness_strength)))
    out = np.logical_or(binary, add)
    out = np.logical_and(out, np.logical_not(remove))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return out


def _build_srt_structure_field(
    working_crop: Image.Image,
    scaled_local_bbox: Sequence[int | float],
    morphology_mask: Image.Image,
    *,
    cls_name: str,
    target_key: str,
    target_spectrum: dict[str, float | int],
    seed: int,
    config: DELGIConfig,
) -> tuple[Image.Image, Image.Image, SRTStructurePlan]:
    """Build a transported defect structure field phi and a source-erasure field."""

    width, height = working_crop.size
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    family = _srt_family_from_class(cls_name)
    source_binary = _estimate_source_structure_mask(working_crop, scaled_local_bbox, morphology_mask)
    label_bbox = _clip_bbox_to_size(scaled_local_bbox, (width, height), min_size=2)
    source_bbox = _robust_binary_bbox(source_binary) or _binary_bbox(source_binary) or _clip_bbox_to_size(scaled_local_bbox, (width, height), min_size=2)
    source_metrics = _structure_metrics(source_binary)

    sx0, sy0, sx1, sy1 = source_bbox
    sw, sh = max(2.0, float(sx1 - sx0)), max(2.0, float(sy1 - sy0))
    source_local_fraction = _mask_local_area_fraction(source_binary, source_bbox)
    target_local_fraction = _srt_target_local_fraction(
        family,
        target_spectrum,
        source_fraction=source_local_fraction,
    )
    target_angle = _orientation_angle_from_spectrum(target_spectrum)
    source_angle = float(source_metrics.get("orientation_deg", 0.0))
    target_elongation = float(np.clip(float(target_spectrum.get("elongation", max(sw / sh, sh / sw))), 1.0, 12.0))
    area_gain = math.sqrt(target_local_fraction / max(0.004, source_local_fraction))
    jitter = float(np.clip(config.srt_scale_jitter, 0.0, 1.0))
    global_scale = float(np.clip(1.0 + (area_gain - 1.0) * 0.45 + rng.normal(0.0, 0.20 * jitter), 0.58, 1.72))

    if family in {"scratch", "crack"}:
        scale_x = float(np.clip(global_scale * rng.uniform(1.05, 1.0 + 0.75 * jitter), 0.62, 2.10))
        scale_y = float(np.clip(global_scale * rng.uniform(0.72, 1.0 + 0.18 * jitter), 0.42, 1.45))
    elif family in {"patch", "scale"}:
        scale_x = float(np.clip(global_scale * rng.uniform(0.82, 1.0 + 0.48 * jitter), 0.55, 1.85))
        scale_y = float(np.clip(global_scale * rng.uniform(0.82, 1.0 + 0.48 * jitter), 0.55, 1.85))
    elif family in {"pitted", "inclusion"}:
        scale_x = float(np.clip(global_scale * rng.uniform(0.72, 1.0 + 0.40 * jitter), 0.48, 1.55))
        scale_y = float(np.clip(global_scale * rng.uniform(0.72, 1.0 + 0.48 * jitter), 0.48, 1.65))
    else:
        scale_x = float(np.clip(global_scale * rng.uniform(0.75, 1.0 + 0.45 * jitter), 0.50, 1.70))
        scale_y = float(np.clip(global_scale * rng.uniform(0.75, 1.0 + 0.45 * jitter), 0.50, 1.70))

    if target_elongation > 1.5 and family in {"scratch", "crack", "inclusion"}:
        if sw >= sh:
            scale_x *= float(np.clip(math.sqrt(target_elongation / max(1.0, sw / sh)), 0.70, 1.35))
        else:
            scale_y *= float(np.clip(math.sqrt(target_elongation / max(1.0, sh / sw)), 0.70, 1.35))

    bbox_jitter = float(np.clip(config.srt_bbox_jitter, 0.0, 1.0))
    label_x0, label_y0, label_x1, label_y1 = label_bbox
    label_w = max(2.0, float(label_x1 - label_x0))
    label_h = max(2.0, float(label_y1 - label_y0))
    if bool(config.srt_bbox_update):
        max_shift_x = max(1.0, bbox_jitter * 0.34 * sw)
        max_shift_y = max(1.0, bbox_jitter * 0.34 * sh)
    else:
        max_shift_x = max(1.0, bbox_jitter * 0.10 * label_w)
        max_shift_y = max(1.0, bbox_jitter * 0.10 * label_h)
    cx = (sx0 + sx1) * 0.5 + float(rng.normal(0.0, max_shift_x))
    cy = (sy0 + sy1) * 0.5 + float(rng.normal(0.0, max_shift_y))
    tw = max(2.0, sw * scale_x)
    th = max(2.0, sh * scale_y)
    if not bool(config.srt_bbox_update):
        cx = float(np.clip(cx, label_x0 + 1.0, label_x1 - 1.0))
        cy = float(np.clip(cy, label_y0 + 1.0, label_y1 - 1.0))
        tw = min(tw, max(2.0, label_w * 0.96))
        th = min(th, max(2.0, label_h * 0.96))
    target_bbox = _clip_bbox_to_size((cx - tw * 0.5, cy - th * 0.5, cx + tw * 0.5, cy + th * 0.5), (width, height), min_size=2)
    if not bool(config.srt_bbox_update):
        target_bbox = _fit_bbox_inside(target_bbox, label_bbox, (width, height), min_size=2)
    prior_binary = np.asarray(morphology_mask.convert("L"), dtype=np.uint8) > 32
    prior_bbox = _binary_bbox(prior_binary) or _clip_bbox_to_size(scaled_local_bbox, (width, height), min_size=2)
    if not bool(config.srt_bbox_update):
        transported, counterfactual_info = _sample_intra_box_counterfactual_structure(
            source_binary=source_binary,
            prior_binary=prior_binary,
            label_bbox=label_bbox,
            family=family,
            target_spectrum=target_spectrum,
            target_fraction=target_local_fraction,
            rng=rng,
            config=config,
        )
        target_bbox = label_bbox
        fallback_structure = _constrain_binary_to_bbox(prior_binary, label_bbox)
    else:
        transported = _warp_binary_to_bbox(
            source_binary,
            source_bbox,
            target_bbox,
            angle_delta=float(target_angle - source_angle),
            output_size=(width, height),
        )
        prior_warped = _warp_binary_to_bbox(
            prior_binary,
            prior_bbox,
            target_bbox,
            angle_delta=float(target_angle - source_angle) * 0.45,
            output_size=(width, height),
        )
        source_keep = float(np.clip(config.srt_source_preservation, 0.0, 1.0))
        if source_keep < 0.5:
            transported = np.logical_or(transported, np.logical_and(prior_warped, rng.random(prior_warped.shape) > source_keep))
        else:
            transported = np.logical_or(transported, np.logical_and(prior_warped, rng.random(prior_warped.shape) > 0.72))

        transported = _add_srt_structural_events(
            transported,
            target_bbox,
            family=family,
            target_spectrum=target_spectrum,
            rng=rng,
            config=config,
        )
        target_roughness = float(np.clip(float(target_spectrum.get("boundary_roughness", 0.75)), 0.0, 3.0))
        roughness_strength = float(np.clip(config.srt_boundary_roughness, 0.0, 1.0)) * float(np.clip(target_roughness / 1.7, 0.25, 1.0))
        transported = _roughen_structure_boundary(transported, rng=rng, roughness_strength=roughness_strength)
        transported = _cap_structure_local_area(
            transported,
            target_bbox,
            target_fraction=target_local_fraction,
            family=family,
            rng=rng,
        )
        counterfactual_info = {
            "counterfactual_generation": "bbox_transport_with_events",
            "counterfactual_structure_iou": float(_mask_iou(transported, source_binary)),
            "active_bbox_label_coverage": float(_bbox_area_fraction(_binary_bbox(transported) or target_bbox, label_bbox)),
            "counterfactual_area_score": 0.0,
            "counterfactual_attempts": 1.0,
        }
        fallback_structure = prior_warped
    if not bool(config.srt_bbox_update):
        lx0, ly0, lx1, ly1 = label_bbox
        label_constrained = np.zeros_like(transported, dtype=bool)
        label_constrained[ly0:ly1, lx0:lx1] = transported[ly0:ly1, lx0:lx1]
        transported = label_constrained
    transported_bbox = _robust_binary_bbox(transported, low_q=1.5, high_q=98.5) or _binary_bbox(transported) or target_bbox
    x0, y0, x1, y1 = transported_bbox
    constrained = np.zeros_like(transported, dtype=bool)
    constrained[y0:y1, x0:x1] = transported[y0:y1, x0:x1]
    transported = constrained
    if int(transported.sum()) < max(5, int(source_binary.sum() * 0.18)):
        transported = fallback_structure if fallback_structure.any() else _constrain_binary_to_bbox(source_binary, label_bbox)
        transported = _cap_structure_local_area(
            transported,
            target_bbox,
            target_fraction=target_local_fraction,
            family=family,
            rng=rng,
        )
        if not bool(config.srt_bbox_update):
            lx0, ly0, lx1, ly1 = label_bbox
            label_constrained = np.zeros_like(transported, dtype=bool)
            label_constrained[ly0:ly1, lx0:lx1] = transported[ly0:ly1, lx0:lx1]
            transported = label_constrained
    transported_metrics = _structure_metrics(transported)
    generated_bbox = _robust_binary_bbox(transported, low_q=1.0, high_q=99.0) or _binary_bbox(transported) or transported_bbox
    transported_local_fraction = _mask_local_area_fraction(transported, generated_bbox)
    s2c_structure = _hdsi_s2c_structure_alignment(
        transported,
        label_bbox,
        target_spectrum=target_spectrum,
        target_fraction=target_local_fraction,
    )
    soft = _soft_structure_mask(transported, roughness=float(transported_metrics.get("boundary_roughness", 0.8)))
    source_soft = _soft_structure_mask(source_binary, roughness=float(source_metrics.get("boundary_roughness", 0.8)))

    iou = _bbox_iou(source_bbox, generated_bbox)
    area_delta = abs(float(transported_metrics["area_fraction"]) - float(source_metrics["area_fraction"]))
    component_delta = abs(int(transported_metrics["component_count"]) - int(source_metrics["component_count"]))
    skeleton_delta = abs(float(transported_metrics["skeleton_length"]) - float(source_metrics["skeleton_length"])) / max(1.0, float(source_metrics["skeleton_length"]))
    transport_score = float(
        np.clip(
            0.36 * (1.0 - iou)
            + 0.24 * np.clip(area_delta / max(0.015, float(source_metrics["area_fraction"])), 0.0, 1.0)
            + 0.18 * np.clip(component_delta / 4.0, 0.0, 1.0)
            + 0.22 * np.clip(skeleton_delta, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    if bool(config.use_hdsi_s2c):
        structure_weight = float(np.clip(config.hdsi_s2c_structure_weight, 0.0, 0.75))
        transport_score = float(
            np.clip(
                (1.0 - structure_weight) * transport_score
                + structure_weight * float(s2c_structure["hdsi_s2c_structure_score"]),
                0.0,
                1.0,
            )
        )
    plan = SRTStructurePlan(
        target_key=target_key,
        cls_name=cls_name,
        seed=int(seed),
        family=family,
        source_bbox_scaled=tuple(int(v) for v in source_bbox),
        transported_bbox_scaled=tuple(int(v) for v in target_bbox),
        generated_bbox_scaled=tuple(int(v) for v in generated_bbox),
        source_area_fraction=float(source_metrics["area_fraction"]),
        transported_area_fraction=float(transported_metrics["area_fraction"]),
        source_local_area_fraction=float(source_local_fraction),
        target_local_area_fraction=float(target_local_fraction),
        transported_local_area_fraction=float(transported_local_fraction),
        source_component_count=int(source_metrics["component_count"]),
        transported_component_count=int(transported_metrics["component_count"]),
        source_skeleton_length=float(source_metrics["skeleton_length"]),
        transported_skeleton_length=float(transported_metrics["skeleton_length"]),
        source_mean_width=float(source_metrics["mean_width"]),
        transported_mean_width=float(transported_metrics["mean_width"]),
        source_boundary_roughness=float(source_metrics["boundary_roughness"]),
        transported_boundary_roughness=float(transported_metrics["boundary_roughness"]),
        source_orientation_deg=float(source_angle),
        target_orientation_deg=float(target_angle),
        bbox_shift_x=float(((generated_bbox[0] + generated_bbox[2]) - (source_bbox[0] + source_bbox[2])) * 0.5),
        bbox_shift_y=float(((generated_bbox[1] + generated_bbox[3]) - (source_bbox[1] + source_bbox[3])) * 0.5),
        bbox_scale_x=float((generated_bbox[2] - generated_bbox[0]) / max(1.0, sw)),
        bbox_scale_y=float((generated_bbox[3] - generated_bbox[1]) / max(1.0, sh)),
        structure_transport_score=transport_score,
        hdsi_s2c_structure_score=float(s2c_structure["hdsi_s2c_structure_score"]),
        hdsi_s2c_area_score=float(s2c_structure["hdsi_s2c_area_score"]),
        hdsi_s2c_orientation_score=float(s2c_structure["hdsi_s2c_orientation_score"]),
        hdsi_s2c_roughness_score=float(s2c_structure["hdsi_s2c_roughness_score"]),
        hdsi_s2c_density_score=float(s2c_structure["hdsi_s2c_density_score"]),
        hdsi_s2c_elongation_score=float(s2c_structure["hdsi_s2c_elongation_score"]),
        counterfactual_structure_iou=float(counterfactual_info.get("counterfactual_structure_iou", _mask_iou(transported, source_binary))),
        active_bbox_label_coverage=float(counterfactual_info.get("active_bbox_label_coverage", _bbox_area_fraction(generated_bbox, label_bbox))),
        counterfactual_generation=str(counterfactual_info.get("counterfactual_generation", "unknown")),
        label_source="structure-field-derived" if bool(config.srt_bbox_update) else "source-label-inherited",
    )
    return _float_mask_image(soft), _float_mask_image(source_soft), plan


def _union_mask_images(*masks: Image.Image) -> Image.Image:
    arrays = [
        np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        for mask in masks
        if mask is not None
    ]
    if not arrays:
        return Image.new("L", (1, 1), 0)
    merged = np.maximum.reduce(arrays)
    return _float_mask_image(np.clip(merged, 0.0, 1.0))


def _scale_bbox_between_sizes(
    bbox: Sequence[int | float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    sx = float(target_size[0]) / max(1.0, float(source_size[0]))
    sy = float(target_size[1]) / max(1.0, float(source_size[1]))
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return _clip_bbox_to_size((x0 * sx, y0 * sy, x1 * sx, y1 * sy), target_size, min_size=1)


def _offset_bbox(
    bbox: Sequence[int | float],
    offset: tuple[int, int],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    ox, oy = offset
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return _clip_bbox_to_size((x0 + ox, y0 + oy, x1 + ox, y1 + oy), size, min_size=1)


def _derive_seca_annotation(
    source_crop: Image.Image,
    generated_crop: Image.Image,
    planned_bbox: Sequence[int | float],
    source_bbox: Sequence[int | float],
    target_mask: Image.Image,
    canvas_mask: Image.Image,
    *,
    cls_name: str,
    config: DELGIConfig,
) -> tuple[tuple[int, int, int, int], Image.Image, dict[str, object]]:
    """Derive labels from agreement between target structure and visible evidence."""

    width, height = source_crop.size
    planned = _clip_bbox_to_size(planned_bbox, (width, height), min_size=2)
    source = _clip_bbox_to_size(source_bbox, (width, height), min_size=2)
    family = _srt_family_from_class(cls_name)

    src = np.asarray(source_crop.convert("L"), dtype=np.float32) / 255.0
    gen = np.asarray(generated_crop.resize((width, height), Image.Resampling.LANCZOS).convert("L"), dtype=np.float32) / 255.0
    target = np.asarray(target_mask.resize((width, height), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255.0
    canvas = np.asarray(canvas_mask.resize((width, height), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255.0

    plan_region = np.zeros((height, width), dtype=np.float32)
    px0, py0, px1, py1 = planned
    plan_region[py0:py1, px0:px1] = 1.0
    source_region = np.zeros((height, width), dtype=np.float32)
    sx0, sy0, sx1, sy1 = source
    source_region[sy0:sy1, sx0:sx1] = 1.0
    empty_mask = Image.new("L", (width, height), 0)

    gate_float = np.maximum.reduce([canvas, target, 0.85 * plan_region, 0.55 * source_region])
    gate = gate_float > 0.025
    if gate.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gate = cv2.dilate(gate.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        gate = np.ones((height, width), dtype=bool)

    def normalize(values: np.ndarray, region: np.ndarray) -> np.ndarray:
        vals = values[region]
        if vals.size == 0:
            return np.zeros_like(values, dtype=np.float32)
        lo = float(np.percentile(vals, 8))
        hi = float(np.percentile(vals, 98))
        if hi <= lo + 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    diff = normalize(np.abs(gen - src), gate)
    gen_dog = np.abs(gen - cv2.GaussianBlur(gen, (0, 0), sigmaX=2.0))
    src_dog = np.abs(src - cv2.GaussianBlur(src, (0, 0), sigmaX=2.0))
    texture = normalize(gen_dog, gate)
    texture_gain = normalize(np.maximum(0.0, gen_dog - 0.55 * src_dog), gate)
    prior_boost = 0.55 + 0.50 * target + 0.22 * canvas + 0.16 * plan_region + 0.08 * source_region
    evidence = (0.52 * diff + 0.30 * texture_gain + 0.18 * texture) * prior_boost
    evidence = cv2.GaussianBlur(evidence.astype(np.float32), (0, 0), sigmaX=0.75)
    evidence = np.where(gate, evidence, 0.0).astype(np.float32)

    values = evidence[gate]
    if values.size == 0 or float(values.max()) <= 1e-6:
        return planned, empty_mask, {
            "label_source": "structure-field-derived",
            "planned_bbox_scaled": [int(v) for v in planned],
            "source_bbox_scaled": [int(v) for v in source],
            "refined": False,
            "seca_pass": False,
            "reason": "no_evidence",
        }

    percentile = 78.0 if family in {"scratch", "crack"} else 82.0
    threshold = max(float(np.percentile(values, percentile)), float(values.mean() + 0.22 * values.std()))
    structure_threshold = 0.11 if family in {"scratch", "crack"} else 0.16
    structure = target >= structure_threshold
    if int(structure.sum()) < 4:
        structure = plan_region > 0.5
    structure_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    structure_guard = cv2.dilate(structure.astype(np.uint8), structure_kernel, iterations=1).astype(bool)
    evidence_candidate = np.logical_and(evidence >= threshold, gate)
    candidate = np.logical_and(evidence_candidate, structure_guard)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if family in {"scratch", "crack"}:
        candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel3).astype(bool)
    else:
        candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel3).astype(bool)
        candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel3).astype(bool)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    components: list[tuple[float, int, int]] = []
    planned_area = max(1, (planned[2] - planned[0]) * (planned[3] - planned[1]))
    source_area = max(1, (source[2] - source[0]) * (source[3] - source[1]))
    min_area = max(6, int(0.00035 * width * height), int(0.010 * min(planned_area, source_area)))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label
        overlap_canvas = float(canvas[comp].mean()) if comp.any() else 0.0
        if overlap_canvas < 0.035:
            continue
        overlap_target = float(target[comp].mean()) if comp.any() else 0.0
        overlap_plan = float(plan_region[comp].mean()) if comp.any() else 0.0
        overlap_source = float(source_region[comp].mean()) if comp.any() else 0.0
        evidence_mean = float(evidence[comp].mean())
        score = evidence_mean * math.sqrt(float(area)) * (
            0.58 + 0.48 * overlap_canvas + 0.42 * overlap_target + 0.18 * overlap_plan + 0.10 * overlap_source
        )
        components.append((float(score), int(label), int(area)))

    if not components:
        evidence_total = max(1, int(evidence_candidate.sum()))
        leakage = float(np.logical_and(evidence_candidate, np.logical_not(structure_guard)).sum() / evidence_total)
        off_structure = np.logical_and(gate, np.logical_not(structure_guard))
        on_mean = float(evidence[structure_guard].mean()) if int(structure_guard.sum()) > 0 else 0.0
        off_mean = float(evidence[off_structure].mean()) if int(off_structure.sum()) > 0 else 0.0
        focus = float(on_mean / (off_mean + 1e-6))
        return planned, empty_mask, {
            "label_source": "structure-field-derived",
            "planned_bbox_scaled": [int(v) for v in planned],
            "source_bbox_scaled": [int(v) for v in source],
            "refined": False,
            "seca_pass": False,
            "reason": "no_connected_evidence",
            "evidence_threshold": float(threshold),
            "seca_coverage": 0.0,
            "seca_leakage": leakage,
            "seca_focus": focus,
        }

    components.sort(reverse=True)
    best_score = components[0][0]
    keep_labels = {label for score, label, _area in components if score >= best_score * (0.42 if family in {"scratch", "crack"} else 0.55)}
    if family not in {"scratch", "crack"} and len(keep_labels) > 4:
        keep_labels = {label for _score, label, _area in components[:4]}
    core_mask = np.logical_and(np.isin(labels, list(keep_labels)), structure_guard)

    # SECA uses a high-confidence core for pass/fail scoring, but detector labels
    # should cover the full visible defect extent. Grow the label mask with lower
    # evidence connected to the trusted core instead of boxing only peak evidence.
    relaxed_percentile = 55.0 if family in {"scratch", "crack"} else 62.0
    relaxed_threshold = max(float(np.percentile(values, relaxed_percentile)), float(threshold) * 0.48)
    relaxed_candidate = np.logical_and.reduce((evidence >= relaxed_threshold, gate, structure_guard))
    if family in {"scratch", "crack"}:
        horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
        relaxed_candidate = np.logical_or(
            cv2.morphologyEx(relaxed_candidate.astype(np.uint8), cv2.MORPH_CLOSE, horizontal).astype(bool),
            cv2.morphologyEx(relaxed_candidate.astype(np.uint8), cv2.MORPH_CLOSE, vertical).astype(bool),
        )
        relaxed_candidate = cv2.dilate(
            relaxed_candidate.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)
    else:
        extent_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        relaxed_candidate = cv2.morphologyEx(relaxed_candidate.astype(np.uint8), cv2.MORPH_CLOSE, extent_kernel).astype(bool)
        relaxed_candidate = cv2.dilate(relaxed_candidate.astype(np.uint8), extent_kernel, iterations=1).astype(bool)
    relaxed_candidate = np.logical_and(relaxed_candidate, gate)

    extent_labels_count, extent_labels, extent_stats, _extent_centroids = cv2.connectedComponentsWithStats(
        relaxed_candidate.astype(np.uint8),
        connectivity=8,
    )
    extent_keep: set[int] = set()
    min_extent_area = max(4, int(0.00018 * width * height), int(0.004 * min(planned_area, source_area)))
    for label in range(1, extent_labels_count):
        area = int(extent_stats[label, cv2.CC_STAT_AREA])
        if area < min_extent_area:
            continue
        comp = extent_labels == label
        core_overlap = float(np.logical_and(comp, core_mask).sum() / max(1, int(core_mask.sum())))
        comp_target = float(target[comp].mean()) if comp.any() else 0.0
        comp_canvas = float(canvas[comp].mean()) if comp.any() else 0.0
        comp_evidence = float(evidence[comp].mean()) if comp.any() else 0.0
        if core_overlap > 0.0 or (
            comp_evidence >= float(threshold) * 0.62 and comp_canvas >= 0.035 and comp_target >= 0.045
        ):
            extent_keep.add(int(label))

    label_mask = np.isin(extent_labels, list(extent_keep)) if extent_keep else core_mask.copy()
    label_mask = np.logical_and(label_mask, structure_guard)
    if int(label_mask.sum()) < int(core_mask.sum()):
        label_mask = np.logical_or(label_mask, core_mask)

    core_bbox = _robust_binary_bbox(core_mask, low_q=1.0, high_q=99.0) or _binary_bbox(core_mask) or planned
    refined = _binary_bbox(label_mask) or core_bbox

    pad_base = 7 if family in {"scratch", "crack"} else 9 if family in {"inclusion", "pitted"} else 12
    bx0, by0, bx1, by1 = refined
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)
    pad_x = max(pad_base, int(round(0.10 * bw)))
    pad_y = max(pad_base, int(round(0.10 * bh)))
    if family in {"scratch", "crack"}:
        pad_x = max(pad_x, int(round(0.04 * max(source[2] - source[0], planned[2] - planned[0]))))
        pad_y = max(pad_y, int(round(0.04 * max(source[3] - source[1], planned[3] - planned[1]))))
    refined = _clip_bbox_to_size((bx0 - pad_x, by0 - pad_y, bx1 + pad_x, by1 + pad_y), (width, height), min_size=2)

    refined_area = max(1, (refined[2] - refined[0]) * (refined[3] - refined[1]))
    max_reasonable = (4.2 if family in {"patch", "scale", "pitted"} else 3.4) * max(planned_area, source_area)
    if refined_area > max_reasonable:
        rx0, ry0, rx1, ry1 = core_bbox
        tight = _clip_bbox_to_size((rx0 - pad_x, ry0 - pad_y, rx1 + pad_x, ry1 + pad_y), (width, height), min_size=2)
        refined = tight if _bbox_area_fraction(tight, planned) <= _bbox_area_fraction(refined, planned) else refined

    structure_area = max(1, int(structure.sum()))
    evidence_total = max(1, int(evidence_candidate.sum()))
    coverage = float(np.logical_and(label_mask, structure).sum() / structure_area)
    leakage = float(np.logical_and(evidence_candidate, np.logical_not(structure_guard)).sum() / evidence_total)
    off_structure = np.logical_and(gate, np.logical_not(structure_guard))
    focus = float(evidence[structure_guard].mean() / (evidence[off_structure].mean() + 1e-6)) if int(off_structure.sum()) > 0 else 3.0
    seca_score = float(
        np.clip(
            0.42 * np.clip(coverage / max(1e-4, float(config.seca_min_coverage)), 0.0, 1.0)
            + 0.34 * np.clip((float(config.seca_max_leakage) - leakage) / max(1e-4, float(config.seca_max_leakage)), 0.0, 1.0)
            + 0.24 * np.clip(focus / max(1e-4, float(config.seca_min_focus) * 1.6), 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    seca_pass = bool(
        coverage >= float(config.seca_min_coverage)
        and leakage <= float(config.seca_max_leakage)
        and focus >= float(config.seca_min_focus)
    )
    confidence = float(np.clip(best_score / max(1e-6, float(values.mean() * math.sqrt(max(1, planned_area)))), 0.0, 3.0))
    info = {
        "label_source": "structure-evidence-aligned" if seca_pass else "structure-field-derived",
        "planned_bbox_scaled": [int(v) for v in planned],
        "source_bbox_scaled": [int(v) for v in source],
        "refined_bbox_scaled": [int(v) for v in refined],
        "core_bbox_scaled": [int(v) for v in core_bbox],
        "label_mask_area": int(label_mask.sum()),
        "core_mask_area": int(core_mask.sum()),
        "refined": bool(seca_pass),
        "seca_pass": bool(seca_pass),
        "seca_score": float(seca_score),
        "seca_coverage": float(coverage),
        "seca_leakage": float(leakage),
        "seca_focus": float(focus),
        "evidence_threshold": float(threshold),
        "evidence_confidence": confidence,
        "component_count": int(len(components)),
        "kept_component_count": int(len(keep_labels)),
        "extent_component_count": int(max(0, extent_labels_count - 1)),
        "kept_extent_component_count": int(len(extent_keep)),
        "refinement_iou_planned": float(_bbox_iou(planned, refined)),
        "refinement_iou_source": float(_bbox_iou(source, refined)),
    }
    mask_image = _float_mask_image(label_mask.astype(np.float32))
    return (refined if seca_pass else planned), mask_image, info


def _quality_domains(
    original_crop: Image.Image,
    generated_crop: Image.Image,
    masks: LGIMaskSet,
) -> dict[str, float]:
    orig = np.asarray(original_crop.convert("RGB"), dtype=np.float32)
    gen = np.asarray(generated_crop.resize(original_crop.size, Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32)
    orig_gray = cv2.cvtColor(np.clip(orig, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gen_gray = cv2.cvtColor(np.clip(gen, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    diff = np.abs(gen - orig).mean(axis=2) / 255.0
    core = np.asarray(masks.core.resize(original_crop.size, Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255.0
    shell = np.asarray(masks.shell.resize(original_crop.size, Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255.0
    out = np.asarray(masks.out.resize(original_crop.size, Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255.0
    core_region = core > 0.12
    shell_region = shell > 0.08
    out_region = out > 0.60
    inside_delta = float(diff[core_region].mean()) if core_region.any() else 0.0
    boundary_delta = float(diff[shell_region].mean()) if shell_region.any() else 0.0
    outside_delta = float(diff[out_region].mean()) if out_region.any() else 0.0
    visible_region = core_region if int(core_region.sum()) >= 4 else (core + shell > 0.08)
    change_fraction = float((diff[visible_region] > 0.025).mean()) if visible_region.any() else 0.0
    if visible_region.any():
        src_vals = orig_gray[visible_region]
        gen_vals = gen_gray[visible_region]
        src_mean = float(src_vals.mean())
        gen_mean = float(gen_vals.mean())
        src_var = float(src_vals.var())
        gen_var = float(gen_vals.var())
        cov = float(((src_vals - src_mean) * (gen_vals - gen_mean)).mean())
        c1 = 0.01**2
        c2 = 0.03**2
        denom = (src_mean * src_mean + gen_mean * gen_mean + c1) * (src_var + gen_var + c2)
        source_similarity = (
            ((2.0 * src_mean * gen_mean + c1) * (2.0 * cov + c2)) / max(1e-8, denom)
        )
        source_similarity = float(np.clip(source_similarity, -1.0, 1.0))
        src_lap = cv2.Laplacian(orig_gray, cv2.CV_32F, ksize=3)
        gen_lap = cv2.Laplacian(gen_gray, cv2.CV_32F, ksize=3)
        source_sharpness = float(np.mean(np.abs(src_lap[visible_region])))
        generated_sharpness = float(np.mean(np.abs(gen_lap[visible_region])))
        sharpness_ratio = float(generated_sharpness / max(1e-6, source_sharpness))
    else:
        source_similarity = 1.0
        source_sharpness = 0.0
        generated_sharpness = 0.0
        sharpness_ratio = 1.0
    background_score = math.exp(-outside_delta / 0.018)
    defect_strength_score = 1.0 - math.exp(-inside_delta / 0.045)
    boundary_score = math.exp(-abs(boundary_delta - inside_delta * 0.35) / 0.055)
    total = 0.36 * background_score + 0.42 * defect_strength_score + 0.22 * boundary_score
    return {
        "total": float(total),
        "background_score": float(background_score),
        "defect_strength_score": float(defect_strength_score),
        "boundary_score": float(boundary_score),
        "inside_delta": float(inside_delta),
        "boundary_delta": float(boundary_delta),
        "outside_delta": float(outside_delta),
        "visible_change_fraction": float(change_fraction),
        "source_similarity": float(source_similarity),
        "source_sharpness": float(source_sharpness),
        "generated_sharpness": float(generated_sharpness),
        "sharpness_ratio": float(sharpness_ratio),
    }


def _build_pseudo_normal_canvas(
    working_crop: Image.Image,
    masks: LGIMaskSet,
    *,
    suppression_strength: float = 0.0,
) -> Image.Image:
    """Remove source defect evidence while retaining local industrial texture."""

    image = np.asarray(working_crop.convert("RGB"), dtype=np.uint8)
    edit = np.asarray(masks.edit.resize(working_crop.size, Image.Resampling.BILINEAR).convert("L"), dtype=np.uint8)
    edit = cv2.dilate(edit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    binary = np.where(edit > 10, 255, 0).astype(np.uint8)
    if int(np.count_nonzero(binary)) == 0:
        return working_crop.copy()

    coverage = float(np.count_nonzero(binary) / max(1, binary.size))
    if coverage > 0.35:
        # Large steel-defect boxes often cover most of the crop; full inpainting
        # creates artificial diagonal fill patterns. Suppress high residuals instead.
        inpainted = cv2.bilateralFilter(image, d=9, sigmaColor=32, sigmaSpace=21)
        smooth = cv2.GaussianBlur(image, (0, 0), sigmaX=2.4)
        inpainted = np.clip(0.55 * inpainted.astype(np.float32) + 0.45 * smooth.astype(np.float32), 0, 255).astype(np.uint8)
    else:
        inpainted = cv2.inpaint(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), binary, 3.0, cv2.INPAINT_TELEA)
        inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    feather = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (0, 0), sigmaX=3.0)
    feather = np.clip(feather[..., None], 0.0, 1.0)
    canvas = image.astype(np.float32) * (1.0 - feather) + inpainted.astype(np.float32) * feather
    strength = float(np.clip(suppression_strength, 0.0, 1.0))
    if strength > 0:
        erased = cv2.bilateralFilter(np.clip(canvas, 0, 255).astype(np.uint8), d=11, sigmaColor=42, sigmaSpace=25)
        low = cv2.GaussianBlur(erased, (0, 0), sigmaX=3.2 + 2.2 * strength)
        erased = np.clip((0.55 - 0.18 * strength) * erased.astype(np.float32) + (0.45 + 0.18 * strength) * low.astype(np.float32), 0, 255)
        canvas = canvas * (1.0 - feather * strength) + erased * (feather * strength)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")


def _neutralize_chroma_if_monochrome(
    generated_crop: Image.Image,
    reference_crop: Image.Image,
    *,
    force: bool = False,
) -> Image.Image:
    ref = np.asarray(reference_crop.resize(generated_crop.size, Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32)
    gen = np.asarray(generated_crop.convert("RGB"), dtype=np.float32)
    ref_chroma = float(np.mean(np.std(ref, axis=2)))
    if not force and ref_chroma > 3.0:
        return generated_crop
    gray = 0.299 * gen[..., 0] + 0.587 * gen[..., 1] + 0.114 * gen[..., 2]
    neutral = np.stack([gray, gray, gray], axis=2)
    return Image.fromarray(np.clip(neutral, 0, 255).astype(np.uint8), mode="RGB")


def _latent_highpass(latents: torch.Tensor) -> torch.Tensor:
    smooth = F.avg_pool2d(latents.float(), kernel_size=3, stride=1, padding=1).to(dtype=latents.dtype)
    return latents - smooth


def _latent_frequency_bands(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = latents.float()
    smooth3 = F.avg_pool2d(data, kernel_size=3, stride=1, padding=1)
    smooth9 = F.avg_pool2d(data, kernel_size=9, stride=1, padding=4)
    low = smooth9
    mid = smooth3 - smooth9
    high = data - smooth3
    return low.to(dtype=latents.dtype), mid.to(dtype=latents.dtype), high.to(dtype=latents.dtype)


def _depthwise_filter(latents: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = latents.shape[1]
    weight = kernel.to(device=latents.device, dtype=latents.dtype).view(1, 1, kernel.shape[0], kernel.shape[1])
    weight = weight.repeat(channels, 1, 1, 1)
    return F.conv2d(latents, weight, padding=(kernel.shape[0] // 2, kernel.shape[1] // 2), groups=channels)


def _apply_orientation_mixture(latents: torch.Tensor, target_spectrum: dict[str, float | int]) -> torch.Tensor:
    hist = [
        float(target_spectrum.get("orientation_0", 0.25)),
        float(target_spectrum.get("orientation_45", 0.25)),
        float(target_spectrum.get("orientation_90", 0.25)),
        float(target_spectrum.get("orientation_135", 0.25)),
    ]
    hist = [max(0.0, value) for value in hist]
    total = sum(hist) or 1.0
    hist = [value / total for value in hist]
    k0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0], [0.0, 0.0, 0.0]]) / 4.0
    k45 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]) / 4.0
    k90 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]) / 4.0
    k135 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]) / 4.0
    kernels = (k0, k45, k90, k135)
    mixed = torch.zeros_like(latents)
    for weight, kernel in zip(hist, kernels):
        mixed = mixed + float(weight) * _depthwise_filter(latents, kernel)
    return mixed


def _blend_fft_phase_with_reference(
    latents: torch.Tensor,
    reference: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6:
        return latents
    data = latents.float()
    ref = reference.to(device=latents.device).float()
    spectrum = torch.fft.fft2(data, dim=(-2, -1))
    ref_spectrum = torch.fft.fft2(ref, dim=(-2, -1))
    amplitude = torch.abs(spectrum)
    source_unit = spectrum / torch.abs(spectrum).clamp_min(1e-6)
    ref_unit = ref_spectrum / torch.abs(ref_spectrum).clamp_min(1e-6)
    blended_unit = (1.0 - strength) * source_unit + strength * ref_unit
    blended_unit = blended_unit / torch.abs(blended_unit).clamp_min(1e-6)
    phased = torch.fft.ifft2(amplitude * blended_unit, dim=(-2, -1)).real
    return phased.to(dtype=latents.dtype)


def _masked_rms_tensor(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.clamp(0.0, 1.0)
    denom = (mask.sum() * values.shape[1]).clamp_min(1e-6)
    return torch.sqrt(((values.float() ** 2) * mask.float()).sum() / denom).to(dtype=values.dtype)


def _masked_mean_tensor(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.clamp(0.0, 1.0)
    denom = (mask.sum() * values.shape[1]).clamp_min(1e-6)
    return ((values.float() * mask.float()).sum() / denom).to(dtype=values.dtype)


def _target_spectrum_distance(a: dict[str, object], b: dict[str, object]) -> float:
    keys = [
        "low_freq_ratio",
        "mid_freq_ratio",
        "high_freq_ratio",
        "orientation_0",
        "orientation_45",
        "orientation_90",
        "orientation_135",
        "polarity",
        "boundary_roughness",
        "component_density",
    ]
    diff = 0.0
    for key in keys:
        av = float(a.get(key, 0.0) or 0.0)
        bv = float(b.get(key, 0.0) or 0.0)
        scale = 3.0 if key in {"boundary_roughness", "component_density"} else 1.0
        diff += abs(av - bv) / scale
    return float(min(1.0, diff / 4.0))


def _spectrum_observation_to_dict(obs: DefectSpectrumObservation) -> dict[str, float | int]:
    orient = [
        float(obs.orientation_0),
        float(obs.orientation_45),
        float(obs.orientation_90),
        float(obs.orientation_135),
    ]
    return {
        "low_freq_ratio": float(obs.low_freq_ratio),
        "mid_freq_ratio": float(obs.mid_freq_ratio),
        "high_freq_ratio": float(obs.high_freq_ratio),
        "orientation_0": float(obs.orientation_0),
        "orientation_45": float(obs.orientation_45),
        "orientation_90": float(obs.orientation_90),
        "orientation_135": float(obs.orientation_135),
        "orientation_bin": int(np.argmax(np.asarray(orient, dtype=np.float32))),
        "polarity": float(obs.polarity),
        "boundary_roughness": float(obs.boundary_roughness),
        "component_density": float(obs.component_density),
        "elongation": float(obs.elongation),
        "area_fraction": float(obs.area_fraction),
        "contrast_ratio": float(obs.contrast_ratio),
        "failure_weight": float(obs.failure_weight),
    }


def _jsonable_manifest_value(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _spectrum_profile_to_dict(profile: DefectSpectrumProfile) -> dict[str, float | int]:
    orient = [
        float(profile.orientation_0),
        float(profile.orientation_45),
        float(profile.orientation_90),
        float(profile.orientation_135),
    ]
    return {
        "low_freq_ratio": float(profile.low_freq_ratio),
        "mid_freq_ratio": float(profile.mid_freq_ratio),
        "high_freq_ratio": float(profile.high_freq_ratio),
        "orientation_0": float(profile.orientation_0),
        "orientation_45": float(profile.orientation_45),
        "orientation_90": float(profile.orientation_90),
        "orientation_135": float(profile.orientation_135),
        "orientation_bin": int(np.argmax(np.asarray(orient, dtype=np.float32))),
        "polarity": float(profile.polarity),
        "boundary_roughness": float(profile.boundary_roughness),
        "component_density": float(profile.component_density),
        "elongation": float(profile.elongation),
        "area_fraction": float(profile.area_fraction),
        "contrast_ratio": float(profile.contrast_ratio),
        "failure_weight": float(profile.failure_weight),
    }


def _safe_log_distance(value: float, reference: float, scale: float) -> float:
    value = max(1e-6, float(value))
    reference = max(1e-6, float(reference))
    return float(np.clip(abs(math.log(value / reference)) / max(1e-6, float(scale)), 0.0, 1.0))


def _hdsi_validity_score(candidate: dict[str, object], profile: DefectSpectrumProfile) -> float:
    profile_spectrum = _spectrum_profile_to_dict(profile)
    spectrum_distance = _target_spectrum_distance(candidate, profile_spectrum)
    area_distance = _safe_log_distance(
        float(candidate.get("area_fraction", profile.area_fraction) or profile.area_fraction),
        float(profile.area_fraction),
        1.65,
    )
    contrast_distance = _safe_log_distance(
        float(candidate.get("contrast_ratio", profile.contrast_ratio) or profile.contrast_ratio),
        float(profile.contrast_ratio),
        1.70,
    )
    elongation_distance = _safe_log_distance(
        float(candidate.get("elongation", profile.elongation) or profile.elongation),
        float(profile.elongation),
        1.85,
    )
    class_distance = float(
        np.clip(
            0.55 * spectrum_distance
            + 0.16 * area_distance
            + 0.15 * contrast_distance
            + 0.14 * elongation_distance,
            0.0,
            1.0,
        )
    )
    return float(np.clip(math.exp(-class_distance / 0.42), 0.0, 1.0))


def _hdsi_hardness_score(
    candidate: dict[str, object],
    profile: DefectSpectrumProfile,
    coverage_counts: Optional[dict[int, int]],
) -> float:
    high_ratio = float(candidate.get("high_freq_ratio", profile.high_freq_ratio) or profile.high_freq_ratio)
    area_fraction = float(candidate.get("area_fraction", profile.area_fraction) or profile.area_fraction)
    contrast_ratio = float(candidate.get("contrast_ratio", profile.contrast_ratio) or profile.contrast_ratio)
    boundary = float(candidate.get("boundary_roughness", profile.boundary_roughness) or profile.boundary_roughness)
    density = float(candidate.get("component_density", profile.component_density) or profile.component_density)
    elongation = float(candidate.get("elongation", profile.elongation) or profile.elongation)
    failure_weight = float(candidate.get("failure_weight", profile.failure_weight) or profile.failure_weight)

    area_hard = float(np.clip(1.0 - area_fraction / max(0.004, 1.35 * float(profile.area_fraction)), 0.0, 1.0))
    contrast_hard = float(
        np.clip(
            (1.12 * float(profile.contrast_ratio) - contrast_ratio) / max(0.10, 1.12 * float(profile.contrast_ratio)),
            0.0,
            1.0,
        )
    )
    high_freq_hard = float(np.clip((high_ratio - float(profile.high_freq_ratio) + 0.08) / 0.46, 0.0, 1.0))
    rough_hard = float(np.clip(max(boundary, float(profile.boundary_roughness)) / 1.85, 0.0, 1.0))
    elongation_hard = float(np.clip((elongation - 1.15) / 5.25, 0.0, 1.0))
    density_hard = float(np.clip(density / 2.70, 0.0, 1.0))

    orientation_bin = int(float(candidate.get("orientation_bin", 0) or 0))
    profile_orients = [
        float(profile.orientation_0),
        float(profile.orientation_45),
        float(profile.orientation_90),
        float(profile.orientation_135),
    ]
    profile_prob = profile_orients[min(max(orientation_bin, 0), 3)]
    prior_rarity = float(np.clip(1.0 - profile_prob / max(max(profile_orients), 1e-6), 0.0, 1.0))
    if coverage_counts:
        max_count = max(int(v) for v in coverage_counts.values()) if coverage_counts else 0
        seen = int(coverage_counts.get(orientation_bin, 0))
        history_rarity = float(np.clip(1.0 - seen / max(1, max_count + 1), 0.0, 1.0))
    else:
        history_rarity = 0.35
    orientation_hard = float(np.clip(0.45 * prior_rarity + 0.55 * history_rarity, 0.0, 1.0))

    return float(
        np.clip(
            0.16 * area_hard
            + 0.18 * contrast_hard
            + 0.16 * high_freq_hard
            + 0.13 * rough_hard
            + 0.11 * elongation_hard
            + 0.09 * density_hard
            + 0.09 * float(np.clip(failure_weight, 0.0, 1.0))
            + 0.08 * orientation_hard,
            0.0,
            1.0,
        )
    )


def _apply_hdsi_tail_intervention(
    candidate: dict[str, float | int],
    *,
    hardness_score: float,
    validity_score: float,
    config: DELGIConfig,
) -> dict[str, float | int]:
    out = dict(candidate)
    tail = float(
        np.clip(
            float(config.hdsi_tail_strength) * float(hardness_score) * float(validity_score),
            0.0,
            1.0,
        )
    )
    if tail <= 1e-6:
        return out

    freq = np.asarray(
        [
            max(0.0, float(out.get("low_freq_ratio", 0.25))),
            max(0.0, float(out.get("mid_freq_ratio", 0.35))),
            max(0.0, float(out.get("high_freq_ratio", 0.40))),
        ],
        dtype=np.float32,
    )
    freq *= np.asarray([1.0 - 0.10 * tail, 1.0 + 0.03 * tail, 1.0 + 0.19 * tail], dtype=np.float32)
    freq = _normalize_vector(freq, fallback=(0.25, 0.35, 0.40))
    out["low_freq_ratio"] = float(freq[0])
    out["mid_freq_ratio"] = float(freq[1])
    out["high_freq_ratio"] = float(freq[2])
    out["boundary_roughness"] = float(np.clip(float(out.get("boundary_roughness", 0.65)) * (1.0 + 0.16 * tail), 0.0, 3.0))
    out["component_density"] = float(np.clip(float(out.get("component_density", 0.75)) * (1.0 + 0.10 * tail), 0.0, 4.0))
    out["elongation"] = float(np.clip(float(out.get("elongation", 1.8)) * (1.0 + 0.08 * tail), 1.0, 12.0))
    out["area_fraction"] = float(np.clip(float(out.get("area_fraction", 0.18)) * (1.0 - 0.06 * tail), 0.002, 0.95))
    out["contrast_ratio"] = float(np.clip(float(out.get("contrast_ratio", 1.5)) * (1.0 - 0.08 * tail), 0.05, 20.0))
    out["hdsi_tail_intervention_strength"] = float(tail)
    return out


def _annotate_hdsi_target_spectrum(
    candidate: dict[str, float | int],
    *,
    profile: DefectSpectrumProfile,
    source_spectrum: dict[str, object] | None,
    history: Sequence[dict[str, object]],
    coverage_counts: Optional[dict[int, int]],
    config: DELGIConfig,
) -> dict[str, float | int]:
    source_distance = _target_spectrum_distance(candidate, source_spectrum) if source_spectrum is not None else 0.50
    history_diversity = _spectrum_diversity_gain({str(k): v for k, v in candidate.items()}, history)
    hardness = _hdsi_hardness_score(candidate, profile, coverage_counts)
    validity = _hdsi_validity_score(candidate, profile)
    validity_floor = float(np.clip(config.hdsi_min_validity_score, 0.0, 1.0))
    source_floor = float(np.clip(config.hdsi_min_source_distance, 0.0, 1.0))
    validity_gate_pass = bool(validity >= validity_floor)
    source_gate_pass = bool(source_spectrum is None or source_distance >= source_floor)
    hardness_weight = max(1e-6, float(config.hdsi_hardness_weight))
    diversity_weight = max(0.0, float(config.hdsi_diversity_weight))
    validity_weight = float(np.clip(config.hdsi_validity_weight, 0.0, 1.0))
    semantic_shift = float(np.clip(0.55 * source_distance + 0.45 * history_diversity, 0.0, 1.0))
    hard_diverse = float(
        np.clip(
            (hardness_weight * hardness + diversity_weight * semantic_shift)
            / max(1e-6, hardness_weight + diversity_weight),
            0.0,
            1.0,
        )
    )
    intervention = float(np.clip(hard_diverse * ((1.0 - validity_weight) + validity_weight * validity), 0.0, 1.0))
    annotated = _apply_hdsi_tail_intervention(
        candidate,
        hardness_score=hardness,
        validity_score=validity,
        config=config,
    )
    annotated.update(
        {
            "hdsi_enabled": True,
            "hdsi_hardness_score": float(hardness),
            "hdsi_validity_score": float(validity),
            "hdsi_source_distance": float(source_distance),
            "hdsi_history_diversity": float(history_diversity),
            "hdsi_intervention_score": float(intervention),
            "hdsi_hard_valid_score": float(hard_diverse),
            "hdsi_validity_floor": float(validity_floor),
            "hdsi_source_distance_floor": float(source_floor),
            "hdsi_validity_gate_pass": bool(validity_gate_pass),
            "hdsi_source_distance_gate_pass": bool(source_gate_pass),
            "hdsi_hard_valid_gate_pass": bool(validity_gate_pass and source_gate_pass),
            "utility_score": float(
                np.clip(
                    max(float(candidate.get("utility_score", 0.0)), 0.36 + 0.64 * intervention),
                    0.0,
                    1.0,
                )
            ),
            "energy_multiplier": float(
                np.clip(
                    float(candidate.get("energy_multiplier", 1.0)) * (1.0 + 0.10 * hardness * validity),
                    0.82,
                    1.45,
                )
            ),
        }
    )
    return annotated


def _sample_distant_target_spectrum(
    prior: DefectSpectrumPrior,
    cls_name: str,
    *,
    source_spectrum: dict[str, object] | None,
    seed: int,
    jitter: float,
    utility_guidance: float,
    coverage_counts: Optional[dict[int, int]],
    history: Sequence[dict[str, object]],
    config: DELGIConfig,
) -> dict[str, float | int]:
    if not config.use_distant_spectrum and not config.use_hdsi:
        target = prior.target_spectrum(
            cls_name,
            seed=seed,
            jitter=jitter,
            utility_guidance=utility_guidance,
            coverage_counts=coverage_counts,
        )
        target["hdsi_enabled"] = False
        return target

    profile = prior.profile(cls_name)
    n_candidates = max(
        2,
        int(config.distant_spectrum_candidates),
        int(config.hdsi_candidates) if config.use_hdsi else 2,
    )
    far_weight = float(np.clip(config.distant_spectrum_weight, 0.0, 1.0))
    best_score = -float("inf")
    best: dict[str, float | int] | None = None
    best_source_distance = 0.0
    best_history_distance = 0.0
    for idx in range(n_candidates):
        candidate = prior.target_spectrum(
            cls_name,
            seed=seed + idx * 1009,
            jitter=jitter,
            utility_guidance=utility_guidance,
            coverage_counts=coverage_counts,
        )
        source_distance = _target_spectrum_distance(candidate, source_spectrum) if source_spectrum is not None else 0.50
        history_distance = _spectrum_diversity_gain({str(k): v for k, v in candidate.items()}, history)
        orientation_bin = int(float(candidate.get("orientation_bin", 0)))
        orientation_count = int((coverage_counts or {}).get(orientation_bin, 0))
        reuse_penalty = 0.035 * orientation_count
        if config.use_hdsi:
            annotated = _annotate_hdsi_target_spectrum(
                candidate,
                profile=profile,
                source_spectrum=source_spectrum,
                history=history,
                coverage_counts=coverage_counts,
                config=config,
            )
            hdsi_score = float(annotated.get("hdsi_intervention_score", 0.0))
            distant_score = far_weight * source_distance + (1.0 - far_weight) * history_distance
            score = 0.72 * hdsi_score + 0.28 * distant_score - reuse_penalty
            validity_floor = float(np.clip(config.hdsi_min_validity_score, 0.0, 1.0))
            source_floor = float(np.clip(config.hdsi_min_source_distance, 0.0, 1.0))
            validity_shortfall = max(0.0, validity_floor - float(annotated.get("hdsi_validity_score", 0.0)))
            source_shortfall = max(0.0, source_floor - float(source_distance))
            if validity_shortfall > 0.0:
                score -= 0.72 * validity_shortfall / max(1e-6, validity_floor)
            if source_shortfall > 0.0:
                score -= 0.38 * source_shortfall / max(1e-6, source_floor)
            if str(config.method_preset).lower().replace("_", "-") == "spec-srt":
                floor = max(1e-4, float(config.spec_srt_min_source_spectrum_distance))
                score -= 0.45 * max(0.0, floor - float(source_distance)) / floor
            candidate = annotated
        else:
            score = far_weight * source_distance + (1.0 - far_weight) * history_distance - reuse_penalty
            candidate["hdsi_enabled"] = False
        if score > best_score:
            best_score = float(score)
            best = candidate
            best_source_distance = float(source_distance)
            best_history_distance = float(history_distance)

    if best is None:
        best = prior.target_spectrum(
            cls_name,
            seed=seed,
            jitter=jitter,
            utility_guidance=utility_guidance,
            coverage_counts=coverage_counts,
        )
    best = dict(best)
    best["source_spectrum_distance"] = float(best_source_distance)
    best["history_spectrum_diversity"] = float(best_history_distance)
    best["distant_spectrum_score"] = float(best_score)
    best["distant_spectrum_candidates"] = int(n_candidates)
    best["hdsi_candidates"] = int(n_candidates) if config.use_hdsi else 0
    return best


def _spectrum_diversity_gain(
    target_spectrum: dict[str, object],
    history: Sequence[dict[str, object]],
) -> float:
    if not history:
        return 1.0
    nearest = min(_target_spectrum_distance(target_spectrum, previous) for previous in history)
    return float(np.clip(0.25 + 0.75 * nearest, 0.0, 1.0))


def _stable_int_from_text(text: str) -> int:
    value = 0
    for idx, char in enumerate(str(text)):
        value = (value + (idx + 1) * ord(char)) & 0xFFFFFFFF
    return int(value)


def _select_residual_donor(
    bank: dict[str, Sequence[DefectSample]],
    sample: DefectSample,
    *,
    seed: int,
) -> DefectSample | None:
    donors = _select_residual_donors(bank, sample, seed=seed, count=1)
    return donors[0] if donors else None


def _select_residual_donors(
    bank: dict[str, Sequence[DefectSample]],
    sample: DefectSample,
    *,
    seed: int,
    count: int,
) -> list[DefectSample]:
    same_class = list(bank.get(sample.cls_name, ()))
    if not same_class:
        return []
    current_key = (_sample_target_key(sample), int(getattr(sample, "object_index", 0)))
    candidates = [
        donor
        for donor in same_class
        if (_sample_target_key(donor), int(getattr(donor, "object_index", 0))) != current_key
    ]
    non_source_candidates = [donor for donor in candidates if donor.source_stem != sample.source_stem]
    if len(non_source_candidates) >= max(1, int(count)):
        candidates = non_source_candidates
    if not candidates:
        candidates = same_class
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    rng.shuffle(candidates)
    requested = max(1, int(count))
    if len(candidates) >= requested:
        return candidates[:requested]
    donors = list(candidates)
    while len(donors) < requested:
        donors.append(candidates[int(rng.integers(0, len(candidates)))])
    return donors


def _select_hdsi_pd_prototype_donors(
    bank: dict[str, Sequence[DefectSample]],
    sample: DefectSample,
    *,
    target_spectrum: dict[str, float | int],
    source_spectrum: dict[str, object] | None,
    spectrum_prior: DefectSpectrumPrior,
    seed: int,
    count: int,
    config: DELGIConfig,
) -> tuple[list[DefectSample], list[dict[str, object]]]:
    same_class = list(bank.get(sample.cls_name, ()))
    if not same_class:
        return [], []
    current_key = (_sample_target_key(sample), int(getattr(sample, "object_index", 0)))
    candidates = [
        donor
        for donor in same_class
        if (_sample_target_key(donor), int(getattr(donor, "object_index", 0))) != current_key
    ]
    if not candidates:
        candidates = same_class
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    rng.shuffle(candidates)
    profile = spectrum_prior.profile(sample.cls_name)
    scored: list[tuple[float, DefectSample, dict[str, object]]] = []
    for donor in candidates[: max(8, min(len(candidates), max(16, int(count) * 8)))]:
        try:
            donor_obs = _estimate_sample_spectrum(
                donor,
                resolution=int(config.resolution),
                dilation_factor=float(config.dilation_factor),
            )
            donor_spectrum = _spectrum_observation_to_dict(donor_obs)
        except Exception:
            donor_spectrum = _spectrum_observation_to_dict(_fallback_spectrum_observation(donor.cls_name))
        target_alignment = math.exp(-_target_spectrum_distance(donor_spectrum, target_spectrum) / 0.34)
        source_distance = _target_spectrum_distance(donor_spectrum, source_spectrum) if source_spectrum is not None else 0.50
        same_source_penalty = 0.42 if donor.source_stem == sample.source_stem else 1.0
        hardness = _hdsi_hardness_score(donor_spectrum, profile, None)
        validity = _hdsi_validity_score(donor_spectrum, profile)
        high_ratio = float(donor_spectrum.get("high_freq_ratio", 0.0))
        roughness = float(np.clip(float(donor_spectrum.get("boundary_roughness", 0.0)) / 2.2, 0.0, 1.0))
        detail_prior = float(np.clip(0.58 * high_ratio + 0.42 * roughness, 0.0, 1.0))
        score = float(
            np.clip(
                (
                    0.31 * hardness
                    + 0.24 * target_alignment
                    + 0.22 * source_distance
                    + 0.14 * validity
                    + 0.09 * detail_prior
                )
                * same_source_penalty,
                0.0,
                1.0,
            )
        )
        scored.append(
            (
                score,
                donor,
                {
                    "prototype_score": score,
                    "prototype_target_alignment": float(target_alignment),
                    "prototype_source_distance": float(source_distance),
                    "prototype_hardness": float(hardness),
                    "prototype_validity": float(validity),
                    "prototype_detail_prior": float(detail_prior),
                    "prototype_same_source_penalty": float(same_source_penalty),
                    "prototype_spectrum": {str(k): _jsonable_manifest_value(v) for k, v in donor_spectrum.items()},
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    requested = max(1, int(count))
    selected = scored[:requested]
    donors = [item[1] for item in selected]
    records = [item[2] for item in selected]
    return donors, records


def _compose_hdsi_pd_prototype_delta(
    prototype_deltas: Sequence[torch.Tensor],
    latent_core: torch.Tensor,
    *,
    target_energy: float,
    target_spectrum: dict[str, float | int],
    seed: int,
    config: DELGIConfig,
) -> tuple[torch.Tensor | None, dict[str, float | int | bool | str]]:
    if not prototype_deltas:
        return None, {"hdsi_pd_enabled": False, "hdsi_pd_reason": "no_prototype_delta"}
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    strength = float(np.clip(config.hdsi_pd_strength, 0.0, 1.0))
    phase_strength = float(np.clip(config.hdsi_pd_phase_strength, 0.0, 1.0))
    weights = rng.uniform(0.82, 1.18, size=len(prototype_deltas)).astype(np.float32)
    weights = weights / max(1e-6, float(weights.sum()))
    low_mix = torch.zeros_like(prototype_deltas[0])
    mid_mix = torch.zeros_like(prototype_deltas[0])
    high_mix = torch.zeros_like(prototype_deltas[0])
    high_energies: list[float] = []
    shape_strength = float(np.clip(0.42 + 0.40 * strength, 0.0, 1.0))
    for weight, delta in zip(weights, prototype_deltas):
        shaped = _shape_residual_to_target_spectrum(delta, target_spectrum, strength=shape_strength)
        low, mid, high = _latent_frequency_bands(shaped)
        low_mix = low_mix + float(weight) * low
        mid_mix = mid_mix + float(weight) * mid
        high_mix = high_mix + float(weight) * high
        high_energies.append(float(_masked_rms_tensor(high, latent_core).detach().cpu().item()))

    detail = mid_mix * 0.58 + high_mix * 1.18
    oriented_detail = _apply_orientation_mixture(detail, target_spectrum)
    phase_detail = _blend_fft_phase_with_reference(detail, oriented_detail, strength=phase_strength)
    detail = (1.0 - phase_strength) * detail + phase_strength * phase_detail
    prototype = 0.18 * low_mix + detail
    prototype_energy = _masked_rms_tensor(_latent_highpass(prototype), latent_core).clamp_min(1e-6)
    target = torch.tensor(float(target_energy), device=prototype.device, dtype=prototype.dtype)
    scale = (target / prototype_energy).clamp(0.30, 3.60)
    prototype = prototype * scale
    return prototype, {
        "hdsi_pd_enabled": True,
        "hdsi_pd_mode": "spectrum_phase_prototype_residual",
        "hdsi_pd_prototype_count": int(len(prototype_deltas)),
        "hdsi_pd_strength": float(strength),
        "hdsi_pd_phase_strength": float(phase_strength),
        "hdsi_pd_phase_operator": "fft_unit_phase_blend",
        "hdsi_pd_spectrum_shape_strength": float(shape_strength),
        "hdsi_pd_weight_max": float(np.max(weights)) if weights.size else 0.0,
        "hdsi_pd_high_energy_mean": float(np.mean(high_energies)) if high_energies else 0.0,
        "hdsi_pd_scale": float(scale.detach().cpu().item()),
        "hdsi_pd_energy": float(_masked_rms_tensor(_latent_highpass(prototype), latent_core).detach().cpu().item()),
    }


def _latent_mask_centroid(mask: torch.Tensor) -> tuple[int, int]:
    data = mask.detach().float()[0, 0].clamp(0.0, 1.0)
    total = data.sum()
    h, w = data.shape[-2:]
    if float(total.detach().cpu().item()) <= 1e-6:
        return h // 2, w // 2
    yy = torch.arange(h, device=data.device, dtype=data.dtype).view(h, 1)
    xx = torch.arange(w, device=data.device, dtype=data.dtype).view(1, w)
    cy = int(torch.round((data * yy).sum() / total).detach().cpu().item())
    cx = int(torch.round((data * xx).sum() / total).detach().cpu().item())
    return cy, cx


def _match_residual_centroid(
    residual: torch.Tensor,
    donor_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_y, src_x = _latent_mask_centroid(donor_mask)
    dst_y, dst_x = _latent_mask_centroid(target_mask)
    shift_y = int(dst_y - src_y)
    shift_x = int(dst_x - src_x)
    return (
        torch.roll(residual, shifts=(shift_y, shift_x), dims=(-2, -1)),
        torch.roll(donor_mask, shifts=(shift_y, shift_x), dims=(-2, -1)),
    )


def _latent_mask_bbox_tensor(mask: torch.Tensor) -> tuple[float, float, float, float] | None:
    data = mask.detach().float()[0, 0].clamp(0.0, 1.0)
    max_value = float(data.max().detach().cpu().item()) if data.numel() else 0.0
    if max_value <= 1e-6:
        return None
    threshold = max(0.035, 0.18 * max_value)
    ys, xs = torch.where(data >= threshold)
    if xs.numel() == 0 or ys.numel() == 0:
        return None
    return (
        float(xs.min().detach().cpu().item()),
        float(ys.min().detach().cpu().item()),
        float(xs.max().detach().cpu().item() + 1),
        float(ys.max().detach().cpu().item() + 1),
    )


def _affine_warp_residual_to_target_mask(
    residual: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | str]]:
    """Warp source residual evidence into the target mask bbox without wraparound."""

    if residual.ndim != 4 or source_mask.ndim != 4 or target_mask.ndim != 4:
        transported, shifted = _match_residual_centroid(residual, source_mask, target_mask)
        return transported * target_mask.clamp(0.0, 1.0), shifted, {"srt_transport_operator": "centroid_roll_fallback"}

    batch, _channels, height, width = residual.shape
    source_gate = source_mask.clamp(0.0, 1.0)
    gated_residual = residual * source_gate
    theta_rows: list[torch.Tensor] = []
    scale_x_values: list[float] = []
    scale_y_values: list[float] = []
    for idx in range(batch):
        src_bbox = _latent_mask_bbox_tensor(source_gate[idx : idx + 1])
        dst_bbox = _latent_mask_bbox_tensor(target_mask[idx : idx + 1])
        if src_bbox is None or dst_bbox is None:
            src_y, src_x = _latent_mask_centroid(source_gate[idx : idx + 1])
            dst_y, dst_x = _latent_mask_centroid(target_mask[idx : idx + 1])
            sx0, sx1 = float(src_x), float(src_x + 1)
            sy0, sy1 = float(src_y), float(src_y + 1)
            tx0, tx1 = float(dst_x), float(dst_x + 1)
            ty0, ty1 = float(dst_y), float(dst_y + 1)
            src_w = dst_w = src_h = dst_h = 1.0
        else:
            sx0, sy0, sx1, sy1 = src_bbox
            tx0, ty0, tx1, ty1 = dst_bbox
            src_w = max(1.0, sx1 - sx0)
            src_h = max(1.0, sy1 - sy0)
            dst_w = max(1.0, tx1 - tx0)
            dst_h = max(1.0, ty1 - ty0)
        src_cx = (sx0 + sx1 - 1.0) * 0.5
        src_cy = (sy0 + sy1 - 1.0) * 0.5
        dst_cx = (tx0 + tx1 - 1.0) * 0.5
        dst_cy = (ty0 + ty1 - 1.0) * 0.5
        src_cx_n = 2.0 * src_cx / max(1.0, float(width - 1)) - 1.0
        src_cy_n = 2.0 * src_cy / max(1.0, float(height - 1)) - 1.0
        dst_cx_n = 2.0 * dst_cx / max(1.0, float(width - 1)) - 1.0
        dst_cy_n = 2.0 * dst_cy / max(1.0, float(height - 1)) - 1.0
        scale_x = float(np.clip(src_w / max(1.0, dst_w), 0.25, 4.0))
        scale_y = float(np.clip(src_h / max(1.0, dst_h), 0.25, 4.0))
        theta_rows.append(
            torch.tensor(
                [
                    [scale_x, 0.0, src_cx_n - scale_x * dst_cx_n],
                    [0.0, scale_y, src_cy_n - scale_y * dst_cy_n],
                ],
                device=residual.device,
                dtype=torch.float32,
            )
        )
        scale_x_values.append(float(dst_w / max(1.0, src_w)))
        scale_y_values.append(float(dst_h / max(1.0, src_h)))

    theta = torch.stack(theta_rows, dim=0)
    grid = F.affine_grid(theta, size=residual.shape, align_corners=True).to(dtype=residual.dtype)
    transported = F.grid_sample(
        gated_residual,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped_mask = F.grid_sample(
        source_gate,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).clamp(0.0, 1.0)
    return transported, warped_mask, {
        "srt_transport_operator": "affine_grid_sample_bbox",
        "srt_affine_target_over_source_scale_x": float(np.mean(scale_x_values)) if scale_x_values else 1.0,
        "srt_affine_target_over_source_scale_y": float(np.mean(scale_y_values)) if scale_y_values else 1.0,
    }


def _structure_control_points_from_mask(mask: torch.Tensor) -> np.ndarray | None:
    data = mask.detach().float()[0, 0].clamp(0.0, 1.0).cpu().numpy()
    if data.size == 0 or float(data.max()) <= 1e-6:
        return None
    threshold = max(0.035, 0.18 * float(data.max()))
    ys, xs = np.where(data >= threshold)
    if xs.size < 3:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bbox_points = np.asarray(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [(x0 + x1) * 0.5, (y0 + y1) * 0.5],
        ],
        dtype=np.float32,
    )
    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    centered = coords - coords.mean(axis=0, keepdims=True)
    if coords.shape[0] >= 3:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
        proj = centered @ axis
        order = np.argsort(proj)
        coords_sorted = coords[order]
    else:
        coords_sorted = coords
    qs = (0.06, 0.18, 0.32, 0.50, 0.68, 0.82, 0.94)
    axis_points = []
    for q in qs:
        idx = int(np.clip(round(q * (coords_sorted.shape[0] - 1)), 0, coords_sorted.shape[0] - 1))
        axis_points.append(coords_sorted[idx])
    points = np.vstack([bbox_points, np.asarray(axis_points, dtype=np.float32)])
    # Keep TPS well-conditioned when a thin mask produces repeated points.
    for idx in range(1, points.shape[0]):
        if float(np.min(np.linalg.norm(points[:idx] - points[idx], axis=1))) < 0.35:
            angle = 2.0 * math.pi * idx / max(1, points.shape[0])
            points[idx, 0] += 0.45 * math.cos(angle)
            points[idx, 1] += 0.45 * math.sin(angle)
    h, w = data.shape
    points[:, 0] = np.clip(points[:, 0], 0.0, max(0.0, float(w - 1)))
    points[:, 1] = np.clip(points[:, 1], 0.0, max(0.0, float(h - 1)))
    return points.astype(np.float32)


def _normalize_control_points(points: np.ndarray, width: int, height: int, device: torch.device) -> torch.Tensor:
    pts = torch.as_tensor(points, device=device, dtype=torch.float32)
    x = 2.0 * pts[:, 0] / max(1.0, float(width - 1)) - 1.0
    y = 2.0 * pts[:, 1] / max(1.0, float(height - 1)) - 1.0
    return torch.stack([x, y], dim=1)


def _tps_grid_from_control_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    src = _normalize_control_points(source_points, width, height, device)
    dst = _normalize_control_points(target_points, width, height, device)
    n = int(dst.shape[0])
    diff = dst[:, None, :] - dst[None, :, :]
    r2 = (diff * diff).sum(dim=-1).clamp_min(1e-6)
    k = r2 * torch.log(r2)
    p = torch.cat([torch.ones((n, 1), device=device), dst], dim=1)
    l = torch.zeros((n + 3, n + 3), device=device, dtype=torch.float32)
    l[:n, :n] = k + torch.eye(n, device=device, dtype=torch.float32) * 1e-3
    l[:n, n:] = p
    l[n:, :n] = p.t()
    y = torch.zeros((n + 3, 2), device=device, dtype=torch.float32)
    y[:n, :] = src
    params = torch.linalg.solve(l, y)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    query = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    qdiff = query[:, None, :] - dst[None, :, :]
    qr2 = (qdiff * qdiff).sum(dim=-1).clamp_min(1e-6)
    u = qr2 * torch.log(qr2)
    features = torch.cat([u, torch.ones((query.shape[0], 1), device=device), query], dim=1)
    mapped = (features @ params).reshape(height, width, 2)
    return mapped.clamp(-1.35, 1.35)


def _tps_warp_residual_to_target_mask(
    residual: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | str]]:
    """Non-rigidly warp source residual using source/target structure control points."""

    if residual.ndim != 4 or source_mask.ndim != 4 or target_mask.ndim != 4:
        return _affine_warp_residual_to_target_mask(residual, source_mask, target_mask)
    batch, _channels, height, width = residual.shape
    source_gate = source_mask.clamp(0.0, 1.0)
    gated_residual = residual * source_gate
    grids: list[torch.Tensor] = []
    point_counts: list[int] = []
    try:
        for idx in range(batch):
            src_points = _structure_control_points_from_mask(source_gate[idx : idx + 1])
            dst_points = _structure_control_points_from_mask(target_mask[idx : idx + 1])
            if src_points is None or dst_points is None or src_points.shape[0] != dst_points.shape[0]:
                transported, warped_mask, info = _affine_warp_residual_to_target_mask(residual, source_mask, target_mask)
                return transported, warped_mask, {**info, "srt_transport_operator": "affine_grid_sample_bbox_fallback"}
            grid = _tps_grid_from_control_points(
                src_points,
                dst_points,
                height=height,
                width=width,
                device=residual.device,
            )
            grids.append(grid)
            point_counts.append(int(src_points.shape[0]))
        full_grid = torch.stack(grids, dim=0).to(dtype=residual.dtype)
        transported = F.grid_sample(
            gated_residual,
            full_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped_mask = F.grid_sample(
            source_gate,
            full_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).clamp(0.0, 1.0)
        return transported, warped_mask, {
            "srt_transport_operator": "tps_structure_flow",
            "srt_tps_control_points": float(np.mean(point_counts)) if point_counts else 0.0,
        }
    except RuntimeError:
        transported, warped_mask, info = _affine_warp_residual_to_target_mask(residual, source_mask, target_mask)
        return transported, warped_mask, {**info, "srt_transport_operator": "affine_grid_sample_bbox_fallback"}


def _shape_residual_to_target_spectrum(
    residual: torch.Tensor,
    target_spectrum: dict[str, float | int],
    *,
    strength: float,
) -> torch.Tensor:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return residual
    low, mid, high = _latent_frequency_bands(residual)
    ratios = [
        max(0.0, float(target_spectrum.get("low_freq_ratio", 0.25))),
        max(0.0, float(target_spectrum.get("mid_freq_ratio", 0.35))),
        max(0.0, float(target_spectrum.get("high_freq_ratio", 0.40))),
    ]
    ratio_sum = sum(ratios) or 1.0
    ratios = [value / ratio_sum for value in ratios]
    shaped = (
        low * float(0.65 + ratios[0])
        + mid * float(0.72 + ratios[1])
        + high * float(0.82 + ratios[2])
    )
    oriented = _apply_orientation_mixture(shaped, target_spectrum)
    shaped = (1.0 - 0.55 * strength) * shaped + (0.55 * strength) * oriented
    source_rms = _masked_rms_tensor(residual, torch.ones_like(residual[:, :1])).clamp_min(1e-6)
    shaped_rms = _masked_rms_tensor(shaped, torch.ones_like(shaped[:, :1])).clamp_min(1e-6)
    shaped = shaped * (source_rms / shaped_rms).clamp(0.25, 4.0)
    return (1.0 - strength) * residual + strength * shaped


def _adapt_drr_residual_to_target(
    donor_delta: torch.Tensor,
    donor_mask: torch.Tensor,
    latent_core: torch.Tensor,
    latent_shell: torch.Tensor,
    *,
    target_spectrum: dict[str, float | int],
    seed: int,
    config: DELGIConfig,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    jitter = float(np.clip(config.structure_jitter, 0.0, 1.0))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    residual = donor_delta
    mask = donor_mask
    flip_x = bool(rng.random() < 0.35 * jitter)
    flip_y = bool(rng.random() < 0.22 * jitter)
    if flip_x:
        residual = torch.flip(residual, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    if flip_y:
        residual = torch.flip(residual, dims=(-2,))
        mask = torch.flip(mask, dims=(-2,))
    residual, mask = _match_residual_centroid(residual, mask, latent_core)
    max_shift = int(round(1 + 6 * jitter))
    shift_y = int(rng.integers(-max_shift, max_shift + 1)) if max_shift > 0 else 0
    shift_x = int(rng.integers(-max_shift, max_shift + 1)) if max_shift > 0 else 0
    if shift_y or shift_x:
        residual = torch.roll(residual, shifts=(shift_y, shift_x), dims=(-2, -1))
        mask = torch.roll(mask, shifts=(shift_y, shift_x), dims=(-2, -1))
    residual = _shape_residual_to_target_spectrum(residual, target_spectrum, strength=0.72 * jitter)
    target_gate = (latent_core + 0.65 * latent_shell).clamp(0.0, 1.0)
    residual = residual * target_gate
    donor_gate = _masked_rms_tensor(mask, latent_core).detach()
    target_gate_rms = _masked_rms_tensor(target_gate, latent_core).detach()
    return residual, {
        "drr_flip_x": flip_x,
        "drr_flip_y": flip_y,
        "drr_shift_x": shift_x,
        "drr_shift_y": shift_y,
        "drr_structure_jitter": jitter,
        "donor_target_gate_overlap": float(donor_gate.cpu().item()),
        "target_gate_rms": float(target_gate_rms.cpu().item()),
    }


def _compose_bandwise_drr_residual(
    residuals: Sequence[torch.Tensor],
    latent_core: torch.Tensor,
    *,
    target_spectrum: dict[str, float | int],
    seed: int,
    config: DELGIConfig,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    if not residuals:
        raise ValueError("At least one residual is required for band-wise recomposition.")
    if len(residuals) == 1:
        return residuals[0], {
            "band_recomposition_enabled": False,
            "band_residual_count": 1,
            "band_recomposition_strength": 0.0,
        }

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    strength = float(np.clip(config.band_recomposition_strength, 0.0, 1.0))
    padded = list(residuals)
    while len(padded) < 3:
        padded.append(padded[-1])

    low, _mid0, _high0 = _latent_frequency_bands(padded[0])
    _low1, mid, _high1 = _latent_frequency_bands(padded[1])
    _low2, _mid2, high = _latent_frequency_bands(padded[2])
    weights = rng.uniform(0.78, 1.22, size=3).astype(np.float32)
    ratios = [
        max(0.0, float(target_spectrum.get("low_freq_ratio", 0.25))),
        max(0.0, float(target_spectrum.get("mid_freq_ratio", 0.35))),
        max(0.0, float(target_spectrum.get("high_freq_ratio", 0.40))),
    ]
    ratio_sum = sum(ratios) or 1.0
    ratios = [value / ratio_sum for value in ratios]
    weights *= np.asarray([0.85 + ratios[0], 0.85 + ratios[1], 0.85 + ratios[2]], dtype=np.float32)

    recomposed = low * float(weights[0]) + mid * float(weights[1]) + high * float(weights[2])
    shaped = _shape_residual_to_target_spectrum(recomposed, target_spectrum, strength=0.55 * strength)
    recomposed = (1.0 - strength) * recomposed + strength * shaped

    source_energy = torch.stack([
        _masked_rms_tensor(_latent_highpass(residual), latent_core).float() for residual in padded[:3]
    ]).mean()
    recomposed_energy = _masked_rms_tensor(_latent_highpass(recomposed), latent_core).float().clamp_min(1e-6)
    recomposed = recomposed * (source_energy / recomposed_energy).to(dtype=recomposed.dtype).clamp(0.35, 3.0)
    return recomposed, {
        "band_recomposition_enabled": True,
        "band_residual_count": int(len(residuals)),
        "band_recomposition_strength": float(strength),
        "band_low_weight": float(weights[0]),
        "band_mid_weight": float(weights[1]),
        "band_high_weight": float(weights[2]),
        "band_energy_before": float(source_energy.detach().cpu().item()),
        "band_energy_after": float(_masked_rms_tensor(_latent_highpass(recomposed), latent_core).detach().cpu().item()),
    }


def _mix_drr_residual(
    delta_raw: torch.Tensor,
    donor_delta: torch.Tensor,
    latent_core: torch.Tensor,
    *,
    target_energy: float,
    step_fraction: float,
    config: DELGIConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    stage = float(np.clip(step_fraction, 0.0, 1.0))
    smooth_stage = stage * stage * (3.0 - 2.0 * stage)
    mix = float(np.clip(config.residual_bank_mix, 0.0, 0.85)) * (0.18 + 0.82 * smooth_stage)
    if mix <= 1e-6:
        return delta_raw, {"drr_mix": 0.0, "drr_donor_scale": 0.0, "drr_donor_energy": 0.0}
    donor_energy = _masked_rms_tensor(_latent_highpass(donor_delta), latent_core).clamp_min(1e-6)
    raw_energy = _masked_rms_tensor(_latent_highpass(delta_raw), latent_core).clamp_min(1e-6)
    target = torch.maximum(
        raw_energy,
        torch.tensor(float(target_energy), device=delta_raw.device, dtype=delta_raw.dtype),
    )
    donor_scale = (target / donor_energy).clamp(0.35, 3.25)
    scaled_donor = donor_delta * donor_scale
    mixed = (1.0 - mix) * delta_raw + mix * scaled_donor
    return mixed, {
        "drr_mix": float(mix),
        "drr_donor_scale": float(donor_scale.detach().cpu().item()),
        "drr_donor_energy": float(donor_energy.detach().cpu().item()),
    }


def _transport_srt_residual_to_structure(
    source_delta: torch.Tensor,
    source_mask: torch.Tensor,
    target_core: torch.Tensor,
    target_shell: torch.Tensor,
    *,
    target_energy: float,
    target_spectrum: dict[str, float | int],
    config: DELGIConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    transported, shifted_source_mask, warp_info = _tps_warp_residual_to_target_mask(
        source_delta,
        source_mask,
        target_core,
    )
    strength = float(np.clip(config.srt_transport_strength, 0.0, 1.0))
    transported = _shape_residual_to_target_spectrum(
        transported,
        target_spectrum,
        strength=max(0.12, 0.78 * strength),
    )
    target_gate = (target_core + 0.62 * target_shell).clamp(0.0, 1.0)
    transported = transported * target_gate
    source_energy = _masked_rms_tensor(_latent_highpass(source_delta), source_mask).clamp_min(1e-6)
    transported_energy = _masked_rms_tensor(_latent_highpass(transported), target_core).clamp_min(1e-6)
    target = torch.tensor(float(target_energy), device=transported.device, dtype=transported.dtype)
    desired_energy = torch.maximum(source_energy, target * float(0.92 + 0.35 * strength))
    scale = (desired_energy / transported_energy).clamp(0.38, 4.40)
    transported = transported * scale
    overlap = _masked_rms_tensor(shifted_source_mask, target_core).detach()
    return transported, {
        "srt_residual_source_energy": float(source_energy.detach().cpu().item()),
        "srt_residual_desired_energy": float(desired_energy.detach().cpu().item()),
        "srt_residual_transported_energy": float(_masked_rms_tensor(_latent_highpass(transported), target_core).detach().cpu().item()),
        "srt_residual_scale": float(scale.detach().cpu().item()),
        "srt_source_target_gate_overlap": float(overlap.cpu().item()),
        **warp_info,
    }


def _mix_srt_residual(
    delta_raw: torch.Tensor,
    transported_delta: torch.Tensor,
    latent_core: torch.Tensor,
    *,
    target_energy: float,
    step_fraction: float,
    config: DELGIConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    stage = float(np.clip(step_fraction, 0.0, 1.0))
    until = float(np.clip(config.srt_regeneration_until, 0.08, 0.88))
    early = float(np.clip((until - stage) / until, 0.0, 1.0))
    early_gate = early * early * (3.0 - 2.0 * early)
    late_floor = float(np.clip(config.srt_late_texture_mix, 0.05, 0.70))
    mix = float(np.clip(config.srt_transport_strength, 0.0, 0.96)) * (
        late_floor + (0.98 - late_floor) * early_gate
    )
    if mix <= 1e-6:
        return delta_raw, {"srt_mix": 0.0, "srt_transported_energy": 0.0}
    transported_energy = _masked_rms_tensor(_latent_highpass(transported_delta), latent_core).clamp_min(1e-6)
    raw_energy = _masked_rms_tensor(_latent_highpass(delta_raw), latent_core).clamp_min(1e-6)
    target = torch.maximum(
        raw_energy,
        torch.tensor(float(target_energy), device=delta_raw.device, dtype=delta_raw.dtype),
    )
    scale = (target / transported_energy).clamp(0.30, 3.50)
    mixed = (1.0 - mix) * delta_raw + mix * transported_delta * scale
    return mixed, {
        "srt_mix": float(mix),
        "srt_early_structure_gate": float(early_gate),
        "srt_transported_scale": float(scale.detach().cpu().item()),
        "srt_transported_energy": float(transported_energy.detach().cpu().item()),
    }


def _apply_srt_structure_regeneration(
    latents: torch.Tensor,
    context_for_step: torch.Tensor,
    transported_delta: torch.Tensor,
    latent_core: torch.Tensor,
    *,
    target_energy: float,
    step_fraction: float,
    config: DELGIConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    stage = float(np.clip(step_fraction, 0.0, 1.0))
    until = float(np.clip(config.srt_regeneration_until, 0.08, 0.88))
    early = float(np.clip((until - stage) / until, 0.0, 1.0))
    early_gate = early * early * (3.0 - 2.0 * early)
    guide = float(np.clip(config.srt_regeneration_strength, 0.0, 0.85)) * early_gate
    if guide <= 1e-6:
        return latents, {"srt_regeneration_gate": 0.0}

    core = latent_core.clamp(0.0, 1.0)
    transported_delta = transported_delta.to(device=latents.device, dtype=latents.dtype)
    transported_energy = _masked_rms_tensor(_latent_highpass(transported_delta), core).clamp_min(1e-6)
    target = torch.tensor(float(target_energy), device=latents.device, dtype=latents.dtype)
    scale = (target / transported_energy).clamp(0.35, 3.80)
    structure_anchor = context_for_step.to(device=latents.device, dtype=latents.dtype) + transported_delta * scale
    guided = latents * (1.0 - core * guide) + structure_anchor * (core * guide)
    return guided, {
        "srt_regeneration_gate": float(guide),
        "srt_regeneration_anchor_scale": float(scale.detach().cpu().item()),
    }


def _apply_hdsi_pd_detail_refinement(
    delta_projected: torch.Tensor,
    prototype_delta: torch.Tensor,
    latent_core: torch.Tensor,
    latent_shell: torch.Tensor,
    *,
    target_energy: float,
    target_spectrum: dict[str, float | int],
    step_fraction: float,
    config: DELGIConfig,
    detail_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not bool(config.use_hdsi_pd):
        return delta_projected, {}
    start = float(np.clip(config.hdsi_pd_detail_start, 0.05, 0.92))
    stage = float(np.clip((float(step_fraction) - start) / max(1e-6, 1.0 - start), 0.0, 1.0))
    gate = stage * stage * (3.0 - 2.0 * stage)
    strength = float(np.clip(config.hdsi_pd_late_detail_strength, 0.0, 1.0)) * gate
    noise_strength = float(np.clip(config.hdsi_pd_late_renoise_strength, 0.0, 0.65)) * gate
    if strength <= 1e-6 and noise_strength <= 1e-6:
        return delta_projected, {"hdsi_pd_detail_gate": float(gate)}

    _ = latent_shell
    core = latent_core.clamp(0.0, 1.0).to(device=delta_projected.device, dtype=delta_projected.dtype)
    detail_gate = core
    prototype_delta = prototype_delta.to(device=delta_projected.device, dtype=delta_projected.dtype)
    _low, mid, high = _latent_frequency_bands(prototype_delta)
    detail = _apply_orientation_mixture(mid * 0.42 + high * 1.28, target_spectrum)
    target = torch.tensor(float(target_energy), device=delta_projected.device, dtype=delta_projected.dtype)
    detail_energy = _masked_rms_tensor(detail, core).clamp_min(1e-6)
    detail_scale = (target / detail_energy).clamp(0.25, 3.25)
    refined = delta_projected + detail_gate * detail * detail_scale * strength
    if detail_noise is not None and noise_strength > 1e-6:
        high_noise = _latent_highpass(detail_noise.to(device=delta_projected.device, dtype=delta_projected.dtype))
        noise_energy = _masked_rms_tensor(high_noise, core).clamp_min(1e-6)
        noise_scale = (target / noise_energy).clamp(0.15, 1.60)
        refined = refined + core * high_noise * noise_scale * noise_strength
    else:
        noise_scale = torch.tensor(0.0, device=delta_projected.device, dtype=delta_projected.dtype)
    return refined, {
        "hdsi_pd_detail_gate": float(gate),
        "hdsi_pd_detail_strength": float(strength),
        "hdsi_pd_detail_energy": float(detail_energy.detach().cpu().item()),
        "hdsi_pd_detail_scale": float(detail_scale.detach().cpu().item()),
        "hdsi_pd_noise_strength": float(noise_strength),
        "hdsi_pd_noise_scale": float(noise_scale.detach().cpu().item()),
        "projected_core_energy": float(_masked_rms_tensor(_latent_highpass(refined), core).detach().cpu().item()),
    }


def _apply_e_srt_evidence_feedback(
    delta_projected: torch.Tensor,
    latent_core: torch.Tensor,
    latent_shell: torch.Tensor,
    latent_out: torch.Tensor,
    *,
    target_energy: float,
    step_fraction: float,
    config: DELGIConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calibrate SRT residuals with latent evidence and background leakage proxies."""

    if not bool(config.use_e_srt):
        return delta_projected, {}
    stage = float(np.clip(step_fraction, 0.0, 1.0))
    late_gate = float(np.clip((stage - 0.10) / 0.72, 0.0, 1.0))
    feedback_gate = late_gate * late_gate * (3.0 - 2.0 * late_gate)
    evidence_strength = float(np.clip(config.e_srt_evidence_strength, 0.0, 1.25)) * feedback_gate
    background_strength = float(np.clip(config.e_srt_background_strength, 0.0, 1.25)) * feedback_gate
    if evidence_strength <= 1e-6 and background_strength <= 1e-6:
        return delta_projected, {"e_srt_feedback_gate": float(feedback_gate)}

    core = latent_core.clamp(0.0, 1.0).to(device=delta_projected.device, dtype=delta_projected.dtype)
    shell = latent_shell.clamp(0.0, 1.0).to(device=delta_projected.device, dtype=delta_projected.dtype)
    out = latent_out.clamp(0.0, 1.0).to(device=delta_projected.device, dtype=delta_projected.dtype)
    structure_gate = (core + 0.42 * shell).clamp(0.0, 1.0)
    high = _latent_highpass(delta_projected)
    core_energy = _masked_rms_tensor(high, core).clamp_min(1e-6)
    shell_energy = _masked_rms_tensor(high, shell).clamp_min(1e-6)
    out_energy = _masked_rms_tensor(high, out).clamp_min(1e-6)
    target = torch.tensor(float(target_energy), device=delta_projected.device, dtype=delta_projected.dtype).clamp_min(1e-6)
    min_core = target * float(np.clip(config.e_srt_min_core_energy_ratio, 0.05, 2.0))
    missing = ((min_core - core_energy) / min_core).clamp(0.0, 1.0)
    leakage_budget = torch.maximum(
        target * float(max(0.025, config.background_leakage * 2.4)),
        torch.tensor(0.018, device=delta_projected.device, dtype=delta_projected.dtype),
    )
    leak = ((out_energy - leakage_budget) / leakage_budget).clamp(0.0, 1.0)
    boost = 1.0 + structure_gate * missing * evidence_strength
    damp = (1.0 - out * leak * background_strength).clamp(0.18, 1.0)
    adjusted = delta_projected * boost * damp
    return adjusted, {
        "e_srt_feedback_gate": float(feedback_gate),
        "e_srt_core_energy": float(core_energy.detach().cpu().item()),
        "e_srt_shell_energy": float(shell_energy.detach().cpu().item()),
        "e_srt_out_energy": float(out_energy.detach().cpu().item()),
        "e_srt_missing_ratio": float(missing.detach().cpu().item()),
        "e_srt_leak_ratio": float(leak.detach().cpu().item()),
        "e_srt_evidence_strength": float(evidence_strength),
        "e_srt_background_strength": float(background_strength),
    }


def _project_defect_energy(
    delta_raw: torch.Tensor,
    latent_core: torch.Tensor,
    latent_shell: torch.Tensor,
    latent_out: torch.Tensor,
    *,
    eta: torch.Tensor,
    target_energy: float,
    config: DELGIConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_core_energy = _masked_rms_tensor(_latent_highpass(delta_raw), latent_core)
    target = torch.tensor(float(target_energy), device=delta_raw.device, dtype=delta_raw.dtype)
    scale = target / raw_core_energy.clamp_min(1e-6)
    scale = scale.clamp(float(config.min_projection_scale), float(config.max_projection_scale))

    core_delta = latent_core * delta_raw * scale
    smooth_core = propagate_background_latents(
        core_delta,
        latent_core.clamp(0.0, 1.0),
        sigma=float(config.boundary_smoothing_sigma),
    )
    shell_eta = (eta * float(config.shell_strength)).clamp(0.0, 1.0)
    shell_delta = latent_shell * ((1.0 - shell_eta) * smooth_core + shell_eta * delta_raw * scale)
    out_gate = scale.clamp(max=1.0) * max(0.0, min(1.0, float(config.background_leakage)))
    out_delta = latent_out * delta_raw * out_gate
    projected = core_delta + shell_delta + out_delta

    projected_core_energy = _masked_rms_tensor(_latent_highpass(projected), latent_core)
    bg_leakage = _masked_rms_tensor(projected, latent_out)
    boundary_energy = _masked_rms_tensor(_latent_highpass(projected), latent_shell)
    trace = {
        "raw_core_energy": float(raw_core_energy.detach().cpu().item()),
        "target_energy": float(target.detach().cpu().item()),
        "projection_scale": float(scale.detach().cpu().item()),
        "projected_core_energy": float(projected_core_energy.detach().cpu().item()),
        "background_leakage": float(bg_leakage.detach().cpu().item()),
        "boundary_energy": float(boundary_energy.detach().cpu().item()),
    }
    return projected, trace


def _project_causal_defect_spectrum(
    delta_raw: torch.Tensor,
    latent_core: torch.Tensor,
    latent_shell: torch.Tensor,
    latent_out: torch.Tensor,
    *,
    eta: torch.Tensor,
    target_energy: float,
    target_spectrum: dict[str, float | int],
    step_fraction: float,
    config: DELGIConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    low, mid, high = _latent_frequency_bands(delta_raw)
    target = torch.tensor(float(target_energy), device=delta_raw.device, dtype=delta_raw.dtype)
    ratios = [
        max(0.0, float(target_spectrum.get("low_freq_ratio", 0.25))),
        max(0.0, float(target_spectrum.get("mid_freq_ratio", 0.35))),
        max(0.0, float(target_spectrum.get("high_freq_ratio", 0.40))),
    ]
    ratio_sum = sum(ratios) or 1.0
    ratios = [value / ratio_sum for value in ratios]
    hdsi_enabled = bool(target_spectrum.get("hdsi_enabled", False))
    hdsi_hardness = float(np.clip(float(target_spectrum.get("hdsi_hardness_score", 0.0)), 0.0, 1.0))
    hdsi_validity = float(np.clip(float(target_spectrum.get("hdsi_validity_score", 1.0)), 0.0, 1.0))
    hdsi_intervention = float(np.clip(float(target_spectrum.get("hdsi_intervention_score", 0.0)), 0.0, 1.0))
    hdsi_tail = (
        float(np.clip(float(config.hdsi_tail_strength) * hdsi_hardness * hdsi_validity, 0.0, 1.0))
        if hdsi_enabled
        else 0.0
    )

    raw_low = _masked_rms_tensor(low, latent_core)
    raw_mid = _masked_rms_tensor(mid, latent_core)
    raw_high = _masked_rms_tensor(high, latent_core)
    low_target = target * float(0.72 * ratios[0] * (1.0 - 0.08 * hdsi_tail))
    mid_target = target * float(1.00 * ratios[1])
    high_target = target * float(1.24 * ratios[2] * (1.0 + 0.18 * hdsi_tail))
    scale_low = (low_target / raw_low.clamp_min(1e-6)).clamp(float(config.min_projection_scale), float(config.max_projection_scale))
    scale_mid = (mid_target / raw_mid.clamp_min(1e-6)).clamp(float(config.min_projection_scale), float(config.max_projection_scale))
    scale_high = (high_target / raw_high.clamp_min(1e-6)).clamp(float(config.min_projection_scale), float(config.max_projection_scale))

    spectral_core = latent_core * (low * scale_low + mid * scale_mid + high * scale_high)
    oriented_core = _apply_orientation_mixture(spectral_core, target_spectrum)
    orient_strength = float(np.clip(config.spectrum_orientation_strength, 0.0, 1.0))
    spectral_core = (1.0 - orient_strength) * spectral_core + orient_strength * oriented_core

    raw_energy_scale = (target / _masked_rms_tensor(_latent_highpass(delta_raw), latent_core).clamp_min(1e-6)).clamp(
        float(config.min_projection_scale),
        float(config.max_projection_scale),
    )
    energy_core = latent_core * delta_raw * raw_energy_scale
    stage = float(np.clip(step_fraction, 0.0, 1.0))
    schedule = float(np.clip((stage - 0.12) / 0.58, 0.0, 1.0))
    projection_strength = float(np.clip(config.spectrum_projection_strength, 0.0, 1.0)) * schedule
    if hdsi_enabled:
        hdsi_boost = 1.0 + float(np.clip(config.hdsi_projection_boost, 0.0, 1.0)) * hdsi_intervention
        projection_strength = float(np.clip(projection_strength * hdsi_boost, 0.0, 1.0))
    else:
        hdsi_boost = 1.0
    core_delta = (1.0 - projection_strength) * energy_core + projection_strength * spectral_core

    target_polarity = float(np.clip(float(target_spectrum.get("polarity", 0.0)), -1.0, 1.0))
    current_mean = _masked_mean_tensor(delta_raw, latent_core)
    current_rms = _masked_rms_tensor(delta_raw, latent_core).clamp_min(1e-6)
    current_polarity = (current_mean / current_rms).clamp(-1.0, 1.0)
    polarity_error = torch.tensor(target_polarity, device=delta_raw.device, dtype=delta_raw.dtype) - current_polarity
    core_delta = core_delta + latent_core * polarity_error * target * 0.055 * projection_strength

    boundary_roughness = float(np.clip(float(target_spectrum.get("boundary_roughness", 0.65)), 0.0, 3.0))
    rough_gate = float(np.clip(boundary_roughness / 1.6 + 0.16 * hdsi_tail, 0.0, 1.0))
    smooth_sigma = max(0.45, float(config.boundary_smoothing_sigma) * (1.0 - 0.35 * rough_gate))
    smooth_core = propagate_background_latents(core_delta, latent_core.clamp(0.0, 1.0), sigma=smooth_sigma)
    shell_eta = (eta * float(config.shell_strength) * (0.78 + 0.46 * rough_gate)).clamp(0.0, 1.0)
    shell_delta = latent_shell * ((1.0 - shell_eta) * smooth_core + shell_eta * core_delta)

    utility_score = float(np.clip(float(target_spectrum.get("utility_score", 0.5)), 0.0, 1.0))
    leakage_budget = max(0.0, min(1.0, float(config.background_leakage))) * (0.80 + 0.35 * (1.0 - utility_score))
    leakage_budget *= 1.0 - 0.12 * hdsi_intervention if hdsi_enabled else 1.0
    out_gate = raw_energy_scale.clamp(max=1.0) * leakage_budget
    out_delta = latent_out * delta_raw * out_gate
    projected = core_delta + shell_delta + out_delta

    projected_low, projected_mid, projected_high = _latent_frequency_bands(projected)
    projected_low_energy = _masked_rms_tensor(projected_low, latent_core)
    projected_mid_energy = _masked_rms_tensor(projected_mid, latent_core)
    projected_high_energy = _masked_rms_tensor(projected_high, latent_core)
    projected_core_energy = _masked_rms_tensor(_latent_highpass(projected), latent_core)
    bg_leakage = _masked_rms_tensor(projected, latent_out)
    boundary_energy = _masked_rms_tensor(_latent_highpass(projected), latent_shell)
    target_freq = torch.tensor(ratios, device=delta_raw.device, dtype=torch.float32)
    projected_freq = torch.stack(
        [
            projected_low_energy.float(),
            projected_mid_energy.float(),
            projected_high_energy.float(),
        ]
    )
    projected_freq = projected_freq / projected_freq.sum().clamp_min(1e-6)
    spectrum_distance = torch.mean(torch.abs(projected_freq - target_freq))
    spectrum_match = torch.exp(-spectrum_distance / 0.18)
    trace = {
        "raw_core_energy": float(_masked_rms_tensor(_latent_highpass(delta_raw), latent_core).detach().cpu().item()),
        "target_energy": float(target.detach().cpu().item()),
        "projection_scale": float(raw_energy_scale.detach().cpu().item()),
        "projection_strength": float(projection_strength),
        "projected_core_energy": float(projected_core_energy.detach().cpu().item()),
        "background_leakage": float(bg_leakage.detach().cpu().item()),
        "boundary_energy": float(boundary_energy.detach().cpu().item()),
        "target_low_ratio": float(ratios[0]),
        "target_mid_ratio": float(ratios[1]),
        "target_high_ratio": float(ratios[2]),
        "projected_low_ratio": float(projected_freq[0].detach().cpu().item()),
        "projected_mid_ratio": float(projected_freq[1].detach().cpu().item()),
        "projected_high_ratio": float(projected_freq[2].detach().cpu().item()),
        "spectrum_match_score": float(spectrum_match.detach().cpu().item()),
        "target_polarity": float(target_polarity),
        "current_polarity": float(current_polarity.detach().cpu().item()),
        "boundary_roughness": float(boundary_roughness),
        "utility_score": float(utility_score),
        "hdsi_enabled": bool(hdsi_enabled),
        "hdsi_hardness_score": float(hdsi_hardness),
        "hdsi_validity_score": float(hdsi_validity),
        "hdsi_intervention_score": float(hdsi_intervention),
        "hdsi_tail_strength": float(hdsi_tail),
        "hdsi_projection_boost": float(hdsi_boost),
    }
    return projected, trace


def _de_lgi_quality(
    base_quality: dict[str, float],
    *,
    energy_target: float,
    energy_actual: float,
    background_leakage: float,
) -> dict[str, float]:
    energy_match = math.exp(-abs(float(energy_actual) - float(energy_target)) / max(0.035, float(energy_target) * 0.45))
    leakage_score = math.exp(-float(background_leakage) / 0.035)
    total = 0.62 * float(base_quality["total"]) + 0.24 * energy_match + 0.14 * leakage_score
    return {
        **base_quality,
        "energy_match_score": float(energy_match),
        "latent_background_leakage_score": float(leakage_score),
        "de_lgi_total": float(total),
    }


def _ucs_de_lgi_quality(
    base_quality: dict[str, float],
    *,
    energy_target: float,
    energy_actual: float,
    background_leakage: float,
    spectrum_match: float,
    utility_score: float,
    diversity_gain: float,
    hdsi_intervention_score: float = 0.0,
    hdsi_validity_score: float = 1.0,
    config: DELGIConfig | None = None,
) -> dict[str, float]:
    energy_match = math.exp(-abs(float(energy_actual) - float(energy_target)) / max(0.035, float(energy_target) * 0.45))
    leakage_score = math.exp(-float(background_leakage) / 0.032)
    spectrum_score = float(np.clip(spectrum_match, 0.0, 1.0))
    utility = float(np.clip(utility_score, 0.0, 1.0))
    diversity = float(np.clip(diversity_gain, 0.0, 1.0))
    hdsi = float(np.clip(hdsi_intervention_score, 0.0, 1.0)) * float(np.clip(hdsi_validity_score, 0.0, 1.0))
    total = (
        0.40 * float(base_quality["total"])
        + 0.16 * energy_match
        + 0.18 * spectrum_score
        + 0.11 * leakage_score
        + 0.09 * utility
        + 0.06 * diversity
    )
    if config is not None and bool(config.use_hdsi):
        total = 0.93 * float(total) + 0.07 * hdsi
    return {
        **base_quality,
        "energy_match_score": float(energy_match),
        "spectrum_match_score": float(spectrum_score),
        "latent_background_leakage_score": float(leakage_score),
        "utility_score": float(utility),
        "diversity_gain_score": float(diversity),
        "hdsi_validity_score": float(np.clip(hdsi_validity_score, 0.0, 1.0)),
        "hdsi_intervention_score": float(np.clip(hdsi_intervention_score, 0.0, 1.0)),
        "hdsi_spectrum_score": float(hdsi),
        "ucs_de_lgi_total": float(total),
        "de_lgi_total": float(total),
    }


def _drr_de_lgi_quality(
    base_quality: dict[str, float],
    *,
    energy_target: float,
    energy_actual: float,
    background_leakage: float,
    spectrum_match: float,
    utility_score: float,
    diversity_gain: float,
    residual_recomposition_score: float,
    hdsi_intervention_score: float = 0.0,
    hdsi_validity_score: float = 1.0,
    config: DELGIConfig,
) -> dict[str, float]:
    ucs_quality = _ucs_de_lgi_quality(
        base_quality,
        energy_target=energy_target,
        energy_actual=energy_actual,
        background_leakage=background_leakage,
        spectrum_match=spectrum_match,
        utility_score=utility_score,
        diversity_gain=diversity_gain,
        hdsi_intervention_score=hdsi_intervention_score,
        hdsi_validity_score=hdsi_validity_score,
        config=config,
    )
    inside_delta = float(base_quality.get("inside_delta", 0.0))
    outside_delta = float(base_quality.get("outside_delta", 0.0))
    delta_target = max(1e-4, float(config.diversity_delta_target))
    outside_budget = max(1e-4, float(config.outside_delta_budget))
    inside_diversity = float(np.clip(1.0 - math.exp(-inside_delta / delta_target), 0.0, 1.0))
    over_strong_penalty = math.exp(-max(0.0, inside_delta - 2.2 * delta_target) / max(0.018, delta_target))
    outside_guard = math.exp(-max(0.0, outside_delta - outside_budget) / max(0.006, outside_budget))
    recomposition = float(np.clip(residual_recomposition_score, 0.0, 1.0))
    structural_score = float(
        np.clip(
            (0.52 * inside_diversity + 0.23 * float(np.clip(diversity_gain, 0.0, 1.0)) + 0.25 * recomposition)
            * over_strong_penalty
            * outside_guard,
            0.0,
            1.0,
        )
    )
    diversity_weight = float(np.clip(config.diversity_weight, 0.0, 0.65))
    total = (1.0 - diversity_weight) * float(ucs_quality["ucs_de_lgi_total"]) + diversity_weight * structural_score
    return {
        **ucs_quality,
        "inside_diversity_score": inside_diversity,
        "outside_guard_score": float(outside_guard),
        "residual_recomposition_score": recomposition,
        "drr_structural_score": structural_score,
        "drr_de_lgi_total": float(total),
        "ucs_de_lgi_total": float(total),
        "de_lgi_total": float(total),
    }


def _structure_to_image_consistency(
    source_crop: Image.Image,
    generated_crop: Image.Image,
    target_structure_mask: Image.Image,
    canvas_mask: Image.Image,
    *,
    config: DELGIConfig,
) -> dict[str, float | bool]:
    source = np.asarray(source_crop.convert("L"), dtype=np.float32) / 255.0
    generated = np.asarray(generated_crop.convert("L").resize(source_crop.size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    target = np.asarray(target_structure_mask.convert("L").resize(source_crop.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    canvas = np.asarray(canvas_mask.convert("L").resize(source_crop.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    phi = target > 0.20
    canvas_bool = canvas > 0.05
    if int(phi.sum()) < 4:
        return {
            "s2i_score": 0.0,
            "s2i_visible_delta": 0.0,
            "s2i_off_phi_delta": 0.0,
            "s2i_ratio": 0.0,
            "s2i_visible_score": 0.0,
            "s2i_focus_score": 0.0,
            "s2i_gate_pass": False,
        }
    kernel = np.ones((5, 5), dtype=np.uint8)
    phi_dilated = cv2.dilate(phi.astype(np.uint8), kernel, iterations=1).astype(bool)
    off_phi = np.logical_and(canvas_bool, np.logical_not(phi_dilated))
    if int(off_phi.sum()) < 4:
        off_phi = np.logical_not(phi_dilated)
    diff = np.abs(generated - source)
    on_delta = float(diff[phi].mean())
    off_delta = float(diff[off_phi].mean()) if int(off_phi.sum()) >= 4 else 0.0
    ratio = float(on_delta / (off_delta + 1e-6))
    visible_target = max(1e-4, float(config.srt_s2i_visible_delta_target))
    min_ratio = max(1e-4, float(config.srt_s2i_min_ratio))
    visible_score = float(np.clip(1.0 - math.exp(-on_delta / visible_target), 0.0, 1.0))
    focus_score = float(np.clip((ratio - 0.55 * min_ratio) / max(1e-4, 1.45 * min_ratio), 0.0, 1.0))
    s2i_score = float(np.clip(0.70 * visible_score + 0.30 * focus_score, 0.0, 1.0))
    return {
        "s2i_score": s2i_score,
        "s2i_visible_delta": on_delta,
        "s2i_off_phi_delta": off_delta,
        "s2i_ratio": ratio,
        "s2i_visible_score": visible_score,
        "s2i_focus_score": focus_score,
        "s2i_gate_pass": bool(on_delta >= 0.70 * visible_target and ratio >= min_ratio),
    }


def _orientation_histogram_from_signal(signal: np.ndarray, region: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    region = np.asarray(region, dtype=bool)
    if int(region.sum()) < 4:
        return np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    gx = cv2.Sobel(signal, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(signal, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    angle = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    centers = np.asarray([0.0, 45.0, 90.0, 135.0], dtype=np.float32)
    hist = []
    weights = mag * region.astype(np.float32)
    for center in centers:
        dist = np.minimum(np.abs(angle - center), 180.0 - np.abs(angle - center))
        hist.append(float(weights[dist <= 22.5].sum()))
    return _normalize_vector(np.asarray(hist, dtype=np.float32), fallback=(0.25, 0.25, 0.25, 0.25))


def _estimate_generated_delta_spectrum(
    source_crop: Image.Image,
    generated_crop: Image.Image,
    target_structure_mask: Image.Image,
    canvas_mask: Image.Image,
) -> dict[str, float | int | bool]:
    source = np.asarray(source_crop.convert("L"), dtype=np.float32) / 255.0
    generated = np.asarray(generated_crop.convert("L").resize(source_crop.size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    target = np.asarray(target_structure_mask.convert("L").resize(source_crop.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    canvas = np.asarray(canvas_mask.convert("L").resize(source_crop.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    phi = target > 0.18
    if int(phi.sum()) < 4:
        return {
            **_spectrum_observation_to_dict(_fallback_spectrum_observation("generated")),
            "valid": False,
            "visible_area_pixels": 0,
        }

    signed = generated - source
    diff = np.abs(signed)
    blur1 = cv2.GaussianBlur(diff, (0, 0), sigmaX=1.1)
    blur7 = cv2.GaussianBlur(diff, (0, 0), sigmaX=7.0)
    low_band = np.abs(blur7)
    mid_band = np.abs(blur1 - blur7)
    high_band = np.abs(diff - blur1)
    energies = np.asarray(
        [
            float(low_band[phi].mean()),
            float(mid_band[phi].mean()),
            float(high_band[phi].mean()),
        ],
        dtype=np.float32,
    )
    freq = _normalize_vector(energies, fallback=(0.25, 0.35, 0.40))

    inside = diff[phi]
    threshold = max(
        float(np.percentile(inside, 64.0)) if inside.size else 0.0,
        float(inside.mean() + 0.10 * inside.std()) if inside.size else 0.0,
        0.0035,
    )
    visible = np.logical_and(phi, diff >= threshold)
    if int(visible.sum()) < max(4, int(0.012 * int(phi.sum()))):
        relaxed = max(float(np.percentile(inside, 48.0)) if inside.size else 0.0, 0.002)
        visible = np.logical_and(phi, diff >= relaxed)
    if int(visible.sum()) < 4:
        visible = phi.copy()

    hist = _orientation_histogram_from_signal(diff, visible)
    polarity = float(np.clip(float(np.mean(signed[phi])) / max(1e-6, _rms(signed[phi])), -1.0, 1.0))
    metrics = _structure_metrics(visible)
    bbox = _robust_binary_bbox(phi, low_q=1.0, high_q=99.0) or _binary_bbox(phi)
    if bbox is None:
        bbox_area = max(1, int(phi.sum()))
    else:
        x0, y0, x1, y1 = bbox
        bbox_area = max(1, int((x1 - x0) * (y1 - y0)))
    area_fraction = float(np.clip(int(visible.sum()) / bbox_area, 0.002, 0.95))
    elongation = _bbox_elongation_from_binary(visible, bbox or (0, 0, source.shape[1], source.shape[0]))
    canvas_bool = canvas > 0.05
    outside = np.logical_and(np.logical_not(canvas_bool), np.logical_not(phi))
    background_energy = _rms(signed[outside]) if int(outside.sum()) >= 8 else max(1e-4, _rms(signed[np.logical_not(phi)]))
    contrast_ratio = float(np.clip(_rms(signed[phi]) / max(1e-4, background_energy), 0.05, 20.0))
    component_area_norm = max(1.0, float(bbox_area) / 4096.0)
    component_density = float(np.clip(float(metrics.get("component_count", 0)) / component_area_norm, 0.0, 4.0))

    return {
        "low_freq_ratio": float(freq[0]),
        "mid_freq_ratio": float(freq[1]),
        "high_freq_ratio": float(freq[2]),
        "orientation_0": float(hist[0]),
        "orientation_45": float(hist[1]),
        "orientation_90": float(hist[2]),
        "orientation_135": float(hist[3]),
        "orientation_bin": int(np.argmax(hist)),
        "polarity": float(polarity),
        "boundary_roughness": float(metrics.get("boundary_roughness", 0.0)),
        "component_density": float(component_density),
        "elongation": float(elongation),
        "area_fraction": float(area_fraction),
        "contrast_ratio": float(contrast_ratio),
        "failure_weight": 0.0,
        "valid": True,
        "visible_area_pixels": int(visible.sum()),
    }


def _hdsi_s2c_consistency(
    source_crop: Image.Image,
    generated_crop: Image.Image,
    target_structure_mask: Image.Image,
    canvas_mask: Image.Image,
    *,
    target_spectrum: dict[str, float | int],
    srt_plan: SRTStructurePlan | None,
    s2i_metrics: dict[str, float | bool],
    config: DELGIConfig,
) -> dict[str, object]:
    generated_spectrum = _estimate_generated_delta_spectrum(
        source_crop,
        generated_crop,
        target_structure_mask,
        canvas_mask,
    )
    tolerance = max(1e-4, float(config.hdsi_s2c_spectrum_tolerance))
    spectral_distance = _target_spectrum_distance(generated_spectrum, target_spectrum)
    spectrum_match = float(math.exp(-spectral_distance / tolerance))

    target_angle = _orientation_angle_from_spectrum(target_spectrum)
    generated_angle = _orientation_angle_from_spectrum(generated_spectrum)
    orientation_score = _angle_match_score(generated_angle, target_angle, scale=45.0)
    area_score = _log_match_score(
        float(generated_spectrum.get("area_fraction", 0.01)),
        float(target_spectrum.get("area_fraction", 0.01)),
        scale=1.10,
    )
    roughness_score = float(
        math.exp(
            -abs(
                float(generated_spectrum.get("boundary_roughness", 0.0))
                - float(target_spectrum.get("boundary_roughness", 0.65))
            )
            / 1.20
        )
    )
    density_score = float(
        math.exp(
            -abs(
                float(generated_spectrum.get("component_density", 0.0))
                - float(target_spectrum.get("component_density", 0.75))
            )
            / 2.20
        )
    )
    contrast_score = _log_match_score(
        float(generated_spectrum.get("contrast_ratio", 1.0)),
        float(target_spectrum.get("contrast_ratio", 1.0)),
        scale=1.35,
    )
    image_spectrum_score = float(
        np.clip(
            0.34 * spectrum_match
            + 0.20 * orientation_score
            + 0.14 * area_score
            + 0.12 * roughness_score
            + 0.10 * density_score
            + 0.10 * contrast_score,
            0.0,
            1.0,
        )
    )
    structure_score = (
        float(srt_plan.hdsi_s2c_structure_score)
        if srt_plan is not None
        else float(image_spectrum_score)
    )
    visible_score = float(np.clip(float(s2i_metrics.get("s2i_score", 0.0)), 0.0, 1.0))
    structure_weight = float(np.clip(config.hdsi_s2c_structure_weight, 0.0, 0.75))
    image_weight = float(np.clip(config.hdsi_s2c_image_weight, 0.0, 0.85))
    if structure_weight + image_weight > 0.92:
        scale = 0.92 / max(1e-6, structure_weight + image_weight)
        structure_weight *= scale
        image_weight *= scale
    visible_weight = max(0.0, 1.0 - structure_weight - image_weight)
    score = float(
        np.clip(
            structure_weight * structure_score
            + image_weight * image_spectrum_score
            + visible_weight * visible_score,
            0.0,
            1.0,
        )
    )
    min_score = float(np.clip(config.hdsi_s2c_min_score, 0.0, 1.0))
    gate_failures: list[str] = []
    if structure_score < 0.72 * min_score:
        gate_failures.append("structure_spectrum_mismatch")
    if image_spectrum_score < min_score:
        gate_failures.append("image_spectrum_mismatch")
    if visible_score < 0.58 * min_score:
        gate_failures.append("weak_visible_structure")
    if not bool(generated_spectrum.get("valid", False)):
        gate_failures.append("invalid_generated_spectrum")
    return {
        "hdsi_s2c_enabled": bool(config.use_hdsi_s2c),
        "hdsi_s2c_score": float(score),
        "hdsi_s2c_structure_score": float(structure_score),
        "hdsi_s2c_image_score": float(image_spectrum_score),
        "hdsi_s2c_spectrum_match": float(spectrum_match),
        "hdsi_s2c_spectral_distance": float(spectral_distance),
        "hdsi_s2c_orientation_score": float(orientation_score),
        "hdsi_s2c_area_score": float(area_score),
        "hdsi_s2c_roughness_score": float(roughness_score),
        "hdsi_s2c_density_score": float(density_score),
        "hdsi_s2c_contrast_score": float(contrast_score),
        "hdsi_s2c_visible_score": float(visible_score),
        "hdsi_s2c_gate_pass": bool(not gate_failures),
        "hdsi_s2c_gate_failures": gate_failures,
        "hdsi_s2c_gate_thresholds": {
            "min_score": float(min_score),
            "structure_min": float(0.72 * min_score),
            "visible_min": float(0.58 * min_score),
        },
        "hdsi_s2c_generated_spectrum": {
            str(k): _jsonable_manifest_value(v) for k, v in generated_spectrum.items()
        },
    }


def _srt_de_lgi_quality(
    base_quality: dict[str, float],
    *,
    energy_target: float,
    energy_actual: float,
    background_leakage: float,
    spectrum_match: float,
    utility_score: float,
    diversity_gain: float,
    structure_transport_score: float,
    hdsi_intervention_score: float = 0.0,
    hdsi_validity_score: float = 1.0,
    structure_novelty: float = 0.0,
    residual_recomposition_score: float = 0.0,
    s2i_score: float = 0.0,
    s2i_visible_delta: float = 0.0,
    s2i_ratio: float = 0.0,
    s2i_gate_pass: bool = True,
    seca_score: float = 0.0,
    seca_gate_pass: bool = True,
    config: DELGIConfig,
) -> dict[str, float]:
    ucs_quality = _ucs_de_lgi_quality(
        base_quality,
        energy_target=energy_target,
        energy_actual=energy_actual,
        background_leakage=background_leakage,
        spectrum_match=spectrum_match,
        utility_score=utility_score,
        diversity_gain=diversity_gain,
        hdsi_intervention_score=hdsi_intervention_score,
        hdsi_validity_score=hdsi_validity_score,
        config=config,
    )
    inside_delta = float(base_quality.get("inside_delta", 0.0))
    outside_delta = float(base_quality.get("outside_delta", 0.0))
    delta_target = max(1e-4, float(config.diversity_delta_target))
    outside_budget = max(1e-4, float(config.outside_delta_budget) * 1.25)
    inside_diversity = float(np.clip(1.0 - math.exp(-inside_delta / delta_target), 0.0, 1.0))
    outside_guard = math.exp(-max(0.0, outside_delta - outside_budget) / max(0.006, outside_budget))
    transport = float(np.clip(structure_transport_score, 0.0, 1.0))
    novelty = float(np.clip(structure_novelty, 0.0, 1.0))
    recomposition = float(np.clip(residual_recomposition_score, 0.0, 1.0))
    s2i = float(np.clip(s2i_score, 0.0, 1.0))
    seca = float(np.clip(seca_score, 0.0, 1.0))
    effective_inside = float(np.clip(inside_diversity * (0.45 + 0.55 * novelty), 0.0, 1.0))
    # Prefer clear structural movement but avoid candidates whose change is only
    # a violent full-canvas edit.
    transport_band = math.exp(-max(0.0, transport - 0.82) / 0.22)
    s2i_gate = 1.0 if bool(s2i_gate_pass) else (0.10 if bool(config.srt_strict_s2i_gate) else 0.58)
    seca_gate = 1.0 if (not bool(config.use_seca) or bool(seca_gate_pass)) else (0.12 if bool(config.seca_strict) else 0.62)
    if bool(config.use_seca):
        structural_mix = (
            0.20 * effective_inside
            + 0.17 * transport
            + 0.13 * novelty
            + 0.11 * recomposition
            + 0.20 * s2i
            + 0.13 * seca
            + 0.04 * float(np.clip(diversity_gain, 0.0, 1.0))
            + 0.02 * outside_guard
        )
    else:
        structural_mix = (
            0.24 * effective_inside
            + 0.19 * transport
            + 0.16 * novelty
            + 0.12 * recomposition
            + 0.23 * s2i
            + 0.04 * float(np.clip(diversity_gain, 0.0, 1.0))
            + 0.02 * outside_guard
        )
    structural_score = float(
        np.clip(
            structural_mix * transport_band * outside_guard,
            0.0,
            1.0,
        )
        * s2i_gate
        * seca_gate
    )
    diversity_weight = float(np.clip(max(config.diversity_weight, 0.28), 0.0, 0.72))
    total = (1.0 - diversity_weight) * float(ucs_quality["ucs_de_lgi_total"]) + diversity_weight * structural_score
    return {
        **ucs_quality,
        "inside_diversity_score": inside_diversity,
        "outside_guard_score": float(outside_guard),
        "structure_transport_score": transport,
        "structure_novelty_score": novelty,
        "effective_inbox_change_score": effective_inside,
        "residual_recomposition_score": recomposition,
        "s2i_score": s2i,
        "s2i_visible_delta": float(s2i_visible_delta),
        "s2i_ratio": float(s2i_ratio),
        "s2i_gate_pass": bool(s2i_gate_pass),
        "seca_score": seca,
        "seca_gate_pass": bool(seca_gate_pass),
        "srt_structural_score": structural_score,
        "srt_de_lgi_total": float(total),
        "ucs_de_lgi_total": float(total),
        "de_lgi_total": float(total),
    }


def _spec_srt_quality(
    base_quality: dict[str, float],
    *,
    cls_name: str = "",
    energy_target: float,
    energy_actual: float,
    background_leakage: float,
    spectrum_match: float,
    utility_score: float,
    diversity_gain: float,
    structure_transport_score: float,
    hdsi_intervention_score: float,
    hdsi_validity_score: float,
    source_spectrum_distance: float,
    s2i_score: float,
    s2i_visible_delta: float,
    s2i_ratio: float,
    s2i_gate_pass: bool,
    config: DELGIConfig,
) -> dict[str, float]:
    ucs_quality = _ucs_de_lgi_quality(
        base_quality,
        energy_target=energy_target,
        energy_actual=energy_actual,
        background_leakage=background_leakage,
        spectrum_match=spectrum_match,
        utility_score=utility_score,
        diversity_gain=diversity_gain,
        hdsi_intervention_score=hdsi_intervention_score,
        hdsi_validity_score=hdsi_validity_score,
        config=config,
    )
    inside_delta = float(base_quality.get("inside_delta", 0.0))
    outside_delta = float(base_quality.get("outside_delta", 0.0))
    change_fraction = float(base_quality.get("visible_change_fraction", 0.0))
    source_similarity = float(base_quality.get("source_similarity", 1.0))
    sharpness_ratio = float(base_quality.get("sharpness_ratio", 0.0))
    min_inside = max(1e-4, float(config.spec_srt_min_inside_delta))
    min_change = max(1e-4, float(config.spec_srt_min_change_fraction))
    max_similarity = float(np.clip(config.spec_srt_max_source_similarity, 0.85, 0.999))
    min_sharpness = max(1e-4, float(config.spec_srt_min_sharpness_ratio))
    cls_key = str(cls_name).lower().replace("_", "-")
    if "scratch" in cls_key:
        min_inside *= 0.62
        min_change *= 0.75
        max_similarity = max(max_similarity, 0.9985)
        min_sharpness *= 0.90
    elif "rolled" in cls_key or "scale" in cls_key:
        min_inside *= 0.75
        min_change *= 0.90
        max_similarity = max(max_similarity, 0.992)
        min_sharpness *= 0.85
    elif "crazing" in cls_key:
        min_sharpness *= 0.68
    elif "patch" in cls_key or "pitted" in cls_key:
        min_inside *= 0.85
        max_similarity = max(max_similarity, 0.990)
    outside_budget = max(1e-4, float(config.outside_delta_budget) * 1.20)
    source_floor = max(1e-4, float(config.spec_srt_min_source_spectrum_distance))

    visible_delta_score = float(np.clip(inside_delta / min_inside, 0.0, 1.25) / 1.25)
    visible_fraction_score = float(np.clip(change_fraction / min_change, 0.0, 1.25) / 1.25)
    copy_margin = max(1e-4, max_similarity - 0.90)
    copy_escape_score = float(np.clip((max_similarity - source_similarity) / copy_margin, 0.0, 1.0))
    sharpness_score = float(np.clip(sharpness_ratio / min_sharpness, 0.0, 1.25) / 1.25)
    outside_guard = float(math.exp(-max(0.0, outside_delta - outside_budget) / max(0.006, outside_budget)))
    transport_score = float(np.clip(structure_transport_score, 0.0, 1.0))
    s2i = float(np.clip(s2i_score, 0.0, 1.0))
    hdsi_score = float(np.clip(hdsi_intervention_score, 0.0, 1.0)) * float(np.clip(hdsi_validity_score, 0.0, 1.0))
    source_distance_score = float(np.clip(source_spectrum_distance / source_floor, 0.0, 1.0))
    spectrum_score = float(np.clip(0.48 * spectrum_match + 0.30 * hdsi_score + 0.22 * source_distance_score, 0.0, 1.0))
    visible_score = float(np.clip(0.42 * visible_delta_score + 0.22 * visible_fraction_score + 0.36 * copy_escape_score, 0.0, 1.0))
    gate_failures: list[str] = []
    if not s2i_gate_pass:
        gate_failures.append("s2i")
    if inside_delta < min_inside:
        gate_failures.append("inside_delta")
    if change_fraction < min_change:
        gate_failures.append("visible_change")
    if source_similarity > max_similarity:
        gate_failures.append("source_copy")
    if sharpness_ratio < min_sharpness:
        gate_failures.append("sharpness")
    if source_spectrum_distance < source_floor:
        gate_failures.append("source_spectrum_distance")
    if outside_delta > outside_budget * 1.85:
        gate_failures.append("outside_delta")
    gate_pass = not gate_failures
    gate = 1.0 if gate_pass else 0.18
    selection = float(
        np.clip(
            (
                0.16 * float(ucs_quality["ucs_de_lgi_total"])
                + 0.18 * s2i
                + 0.22 * visible_score
                + 0.14 * sharpness_score
                + 0.08 * outside_guard
                + 0.12 * spectrum_score
                + 0.10 * transport_score
            )
            * gate,
            0.0,
            1.0,
        )
    )
    return {
        **ucs_quality,
        "spec_srt_visible_delta_score": float(visible_delta_score),
        "spec_srt_visible_fraction_score": float(visible_fraction_score),
        "spec_srt_copy_escape_score": float(copy_escape_score),
        "spec_srt_sharpness_score": float(sharpness_score),
        "spec_srt_outside_guard_score": float(outside_guard),
        "spec_srt_spectrum_score": float(spectrum_score),
        "spec_srt_transport_score": float(transport_score),
        "source_spectrum_distance": float(source_spectrum_distance),
        "spec_srt_source_distance_floor": float(source_floor),
        "spec_srt_source_distance_score": float(source_distance_score),
        "spec_srt_gate_thresholds": {
            "inside_delta": float(min_inside),
            "visible_change_fraction": float(min_change),
            "max_source_similarity": float(max_similarity),
            "sharpness_ratio": float(min_sharpness),
            "outside_delta": float(outside_budget * 1.85),
            "source_spectrum_distance": float(source_floor),
        },
        "spec_srt_gate_margins": {
            "inside_delta": float(inside_delta - min_inside),
            "visible_change_fraction": float(change_fraction - min_change),
            "source_similarity": float(max_similarity - source_similarity),
            "sharpness_ratio": float(sharpness_ratio - min_sharpness),
            "outside_delta": float(outside_budget * 1.85 - outside_delta),
            "source_spectrum_distance": float(source_spectrum_distance - source_floor),
        },
        "spec_srt_gate_failures": gate_failures,
        "spec_srt_gate_pass": bool(gate_pass),
        "spec_srt_selection_score": float(selection),
        "s2i_score": s2i,
        "s2i_visible_delta": float(s2i_visible_delta),
        "s2i_ratio": float(s2i_ratio),
        "s2i_gate_pass": bool(s2i_gate_pass),
        "structure_transport_score": float(transport_score),
        "outside_guard_score": float(outside_guard),
        "de_lgi_total": float(selection),
        "ucs_de_lgi_total": float(selection),
        "srt_de_lgi_total": float(selection),
    }


def _hdsi_s2c_quality(
    quality: dict[str, object],
    *,
    cls_name: str,
    s2c_metrics: dict[str, object],
    config: DELGIConfig,
) -> dict[str, object]:
    """Rank HDSI-S2C candidates by target-spectrum realization in phi' and pixels."""

    base_score = float(np.clip(float(quality.get("srt_de_lgi_total", quality.get("de_lgi_total", 0.0))), 0.0, 1.0))
    s2c_score = float(np.clip(float(s2c_metrics.get("hdsi_s2c_score", 0.0)), 0.0, 1.0))
    structure_score = float(np.clip(float(s2c_metrics.get("hdsi_s2c_structure_score", 0.0)), 0.0, 1.0))
    image_score = float(np.clip(float(s2c_metrics.get("hdsi_s2c_image_score", 0.0)), 0.0, 1.0))
    spectrum_match = float(np.clip(float(s2c_metrics.get("hdsi_s2c_spectrum_match", 0.0)), 0.0, 1.0))
    visible_score = float(np.clip(float(s2c_metrics.get("hdsi_s2c_visible_score", quality.get("s2i_score", 0.0))), 0.0, 1.0))
    hdsi_score = float(np.clip(float(quality.get("hdsi_intervention_score", 0.0)), 0.0, 1.0)) * float(
        np.clip(float(quality.get("hdsi_validity_score", 1.0)), 0.0, 1.0)
    )
    inside_delta = float(quality.get("inside_delta", 0.0))
    outside_delta = float(quality.get("outside_delta", 0.0))
    source_similarity = float(quality.get("source_similarity", 1.0))
    source_distance = float(quality.get("source_spectrum_distance", 0.0))

    max_similarity = float(np.clip(config.e_srt_max_source_similarity, 0.90, 0.9995))
    min_inside = max(1e-4, float(config.e_srt_min_visible_delta))
    max_outside = max(1e-4, float(config.e_srt_max_outside_delta))
    cls_key = str(cls_name).lower().replace("_", "-")
    if "scratch" in cls_key:
        max_similarity = max(max_similarity, 0.9945)
        min_inside *= 0.76
    elif "inclusion" in cls_key or "pitted" in cls_key:
        max_similarity = max(max_similarity, 0.9935)
        min_inside *= 0.86
    elif "rolled" in cls_key or "scale" in cls_key:
        max_similarity = max(max_similarity, 0.9940)
        min_inside *= 0.88

    copy_margin = max(1e-4, max_similarity - 0.90)
    copy_escape = float(np.clip((max_similarity - source_similarity) / copy_margin, 0.0, 1.0))
    visible_delta_score = float(np.clip(inside_delta / min_inside, 0.0, 1.35) / 1.35)
    outside_guard = float(math.exp(-max(0.0, outside_delta - max_outside) / max(0.006, max_outside)))
    source_distance_score = float(
        np.clip(
            source_distance / max(1e-4, float(config.spec_srt_min_source_spectrum_distance)),
            0.0,
            1.0,
        )
    )
    hard_valid_realization = float(
        np.clip(
            0.36 * s2c_score
            + 0.19 * image_score
            + 0.14 * structure_score
            + 0.11 * visible_score
            + 0.08 * copy_escape
            + 0.07 * source_distance_score
            + 0.05 * hdsi_score,
            0.0,
            1.0,
        )
    )
    selection = float(
        np.clip(
            0.58 * hard_valid_realization
            + 0.22 * base_score
            + 0.10 * visible_delta_score
            + 0.06 * outside_guard
            + 0.04 * spectrum_match,
            0.0,
            1.0,
        )
    )

    gate_failures = list(s2c_metrics.get("hdsi_s2c_gate_failures", []) or [])
    if inside_delta < 0.68 * min_inside:
        gate_failures.append("weak_visible_delta")
    if source_similarity > max_similarity:
        gate_failures.append("source_copy")
    if outside_delta > max_outside * 1.32:
        gate_failures.append("outside_delta")
    gate_failures = list(dict.fromkeys(str(reason) for reason in gate_failures))
    gate_pass = not gate_failures
    if not gate_pass and bool(config.hdsi_s2c_strict_gate):
        selection *= 0.20
    elif not gate_pass:
        selection *= 0.72

    return {
        **quality,
        **s2c_metrics,
        "hdsi_s2c_hard_valid_realization_score": float(hard_valid_realization),
        "hdsi_s2c_copy_escape_score": float(copy_escape),
        "hdsi_s2c_visible_delta_score": float(visible_delta_score),
        "hdsi_s2c_outside_guard_score": float(outside_guard),
        "hdsi_s2c_source_distance_score": float(source_distance_score),
        "hdsi_s2c_gate_failures": gate_failures,
        "hdsi_s2c_gate_pass": bool(gate_pass),
        "hdsi_s2c_selection_score": float(selection),
        "de_lgi_total": float(selection),
        "ucs_de_lgi_total": float(selection),
        "srt_de_lgi_total": float(selection),
    }


def _e_srt_hdsi_pd_quality(
    quality: dict[str, float],
    *,
    cls_name: str,
    config: DELGIConfig,
) -> dict[str, float]:
    """Rank E-SRT+HDSI-PD candidates by useful visible intervention, not prototype detail alone."""

    inside_delta = float(quality.get("inside_delta", 0.0))
    outside_delta = float(quality.get("outside_delta", 0.0))
    source_similarity = float(quality.get("source_similarity", 1.0))
    visible_fraction = float(quality.get("visible_change_fraction", 0.0))
    s2i_score = float(np.clip(quality.get("s2i_score", 0.0), 0.0, 1.0))
    s2i_ratio = float(max(0.0, quality.get("s2i_ratio", 0.0)))
    pd_score = float(np.clip(quality.get("hdsi_pd_selection_score", 0.0), 0.0, 1.0))
    base_score = float(np.clip(quality.get("srt_de_lgi_total", quality.get("de_lgi_total", 0.0)), 0.0, 1.0))
    structure_novelty = float(np.clip(quality.get("structure_novelty_score", quality.get("counterfactual_structure_novelty", 0.0)), 0.0, 1.0))
    recomposition = float(np.clip(quality.get("residual_recomposition_score", 0.0), 0.0, 1.0))
    spectrum_diversity = float(np.clip(quality.get("diversity_gain_score", quality.get("diversity_gain", 0.0)), 0.0, 1.0))

    min_inside = max(1e-4, float(config.e_srt_min_visible_delta))
    max_similarity = float(np.clip(config.e_srt_max_source_similarity, 0.90, 0.9995))
    max_outside = max(1e-4, float(config.e_srt_max_outside_delta))
    min_ratio = max(1e-4, float(config.e_srt_min_s2i_ratio))
    min_novelty = float(np.clip(config.e_srt_min_novelty_score, 0.02, 0.85))
    cls_key = str(cls_name).lower().replace("_", "-")
    if "scratch" in cls_key:
        min_inside *= 0.78
        max_similarity = max(max_similarity, 0.9940)
        min_ratio *= 0.72
        min_novelty *= 0.80
    elif "inclusion" in cls_key or "pitted" in cls_key:
        min_inside *= 0.86
        max_similarity = max(max_similarity, 0.9935)
        min_ratio *= 0.86
        min_novelty *= 0.88
    elif "rolled" in cls_key or "scale" in cls_key:
        min_inside *= 0.90
        max_similarity = max(max_similarity, 0.9940)
        min_ratio *= 0.82
        min_novelty *= 0.86

    visible_delta_score = float(np.clip(inside_delta / min_inside, 0.0, 1.35) / 1.35)
    visible_fraction_score = float(np.clip(visible_fraction / 0.060, 0.0, 1.35) / 1.35)
    copy_margin = max(1e-4, max_similarity - 0.90)
    copy_escape_score = float(np.clip((max_similarity - source_similarity) / copy_margin, 0.0, 1.0))
    outside_guard = float(math.exp(-max(0.0, outside_delta - max_outside) / max(0.006, max_outside)))
    focus_score = float(np.clip(s2i_ratio / min_ratio, 0.0, 1.35) / 1.35)
    novelty_score = float(
        np.clip(
            0.42 * structure_novelty
            + 0.24 * recomposition
            + 0.20 * spectrum_diversity
            + 0.14 * copy_escape_score,
            0.0,
            1.0,
        )
    )
    useful_change_score = float(
        np.clip(
            0.28 * visible_delta_score
            + 0.10 * visible_fraction_score
            + 0.15 * copy_escape_score
            + 0.15 * focus_score
            + 0.08 * s2i_score
            + 0.06 * pd_score
            + 0.05 * outside_guard
            + 0.13 * novelty_score,
            0.0,
            1.0,
        )
    )
    selection = float(np.clip(0.64 * useful_change_score + 0.20 * base_score + 0.16 * novelty_score, 0.0, 1.0))
    gate_failures: list[str] = []
    if inside_delta < 0.72 * min_inside:
        gate_failures.append("weak_visible_delta")
    if source_similarity > max_similarity:
        gate_failures.append("source_copy")
    if outside_delta > max_outside * 1.28:
        gate_failures.append("outside_delta")
    if s2i_ratio < 0.62 * min_ratio and inside_delta < min_inside:
        gate_failures.append("weak_structure_focus")
    if novelty_score < min_novelty and source_similarity > max_similarity - 0.010:
        gate_failures.append("low_novelty")

    return {
        **quality,
        "e_srt_hdsi_pd_visible_delta_score": float(visible_delta_score),
        "e_srt_hdsi_pd_visible_fraction_score": float(visible_fraction_score),
        "e_srt_hdsi_pd_copy_escape_score": float(copy_escape_score),
        "e_srt_hdsi_pd_outside_guard_score": float(outside_guard),
        "e_srt_hdsi_pd_focus_score": float(focus_score),
        "e_srt_hdsi_pd_structure_novelty_score": float(structure_novelty),
        "e_srt_hdsi_pd_recomposition_score": float(recomposition),
        "e_srt_hdsi_pd_spectrum_diversity_score": float(spectrum_diversity),
        "e_srt_hdsi_pd_novelty_score": float(novelty_score),
        "e_srt_hdsi_pd_useful_change_score": float(useful_change_score),
        "e_srt_hdsi_pd_gate_failures": gate_failures,
        "e_srt_hdsi_pd_gate_pass": bool(not gate_failures),
        "e_srt_hdsi_pd_gate_thresholds": {
            "inside_delta": float(0.72 * min_inside),
            "target_inside_delta": float(min_inside),
            "max_source_similarity": float(max_similarity),
            "max_outside_delta": float(max_outside * 1.28),
            "target_s2i_ratio": float(min_ratio),
            "min_novelty_score": float(min_novelty),
        },
        "e_srt_hdsi_pd_selection_score": float(selection),
        "de_lgi_total": float(selection),
        "ucs_de_lgi_total": float(selection),
        "srt_de_lgi_total": float(selection),
    }


class ContextLGIGenerator(ReferenceCMDPGenerator):
    """Context-conditioned latent generative inpainting without residual fields."""

    def generate_lgi(
        self,
        samples: Sequence[DefectSample],
        output_dir: Path,
        *,
        domain: str,
        reference_samples: Optional[Sequence[DefectSample]] = None,
        config: ContextLGIConfig | None = None,
    ) -> list[Path]:
        cfg = config or ContextLGIConfig()
        pipe = self._load_pipe()
        device = pipe.unet.device
        dtype = self._dtype()
        output_dir = Path(output_dir)
        image_dir = output_dir / "images"
        artifacts_dir = output_dir / "artifacts" / "context_lgi"
        candidate_dir = artifacts_dir / "candidates"
        crop_dir = artifacts_dir / "expanded_crops"
        pseudo_dir = artifacts_dir / "pseudo_normal_crops"
        gen_crop_dir = artifacts_dir / "generated_expanded_crops"
        mask_dir = artifacts_dir / "masks"
        for path in (image_dir, artifacts_dir, candidate_dir, crop_dir, pseudo_dir, gen_crop_dir, mask_dir):
            path.mkdir(parents=True, exist_ok=True)

        prior = MorphologyPrior.from_samples(reference_samples or samples)
        prior.write_manifest(artifacts_dir / "morphology_prior.json")
        pipe.scheduler.set_timesteps(int(cfg.num_inference_steps), device=device)
        do_cfg = cfg.guidance_scale > 1.0
        prompt_cache: dict[str, torch.Tensor] = {}
        prompts_by_class: dict[str, str] = {}
        rows: list[dict[str, object]] = []
        outputs: list[Path] = []

        for sample_index, sample in enumerate(track(list(samples), description="Context-LGI latent inpainting")):
            target_key = _sample_target_key(sample)
            prompt = reference_caption(domain, sample.cls_name)
            prompts_by_class.setdefault(sample.cls_name, prompt)
            prompt_embeds = prompt_cache.get(prompt)
            if prompt_embeds is None:
                prompt_embeds = self._encode_prompt(pipe, prompt, cfg.guidance_scale)
                prompt_cache[prompt] = prompt_embeds

            original = Image.open(sample.image_path).convert("RGB")
            expanded = expand_bbox(sample.bbox, original.size, cfg.dilation_factor)
            expanded_crop = original.crop(expanded)
            local_bbox = bbox_inside_crop(sample.bbox, expanded)
            working_crop = expanded_crop.resize((cfg.resolution, cfg.resolution), Image.Resampling.LANCZOS)
            scaled_local_bbox = scale_bbox_to_size(local_bbox, expanded_crop.size, working_crop.size)
            crop_dir.joinpath(f"{target_key}.png").parent.mkdir(parents=True, exist_ok=True)
            working_crop.save(crop_dir / f"{target_key}.png")
            try:
                source_spectrum = _spectrum_observation_to_dict(
                    _estimate_sample_spectrum(
                        sample,
                        resolution=int(cfg.resolution),
                        dilation_factor=float(cfg.dilation_factor),
                    )
                )
            except Exception:
                source_spectrum = _spectrum_observation_to_dict(_fallback_spectrum_observation(sample.cls_name))

            best: tuple[float, Image.Image, Image.Image, dict[str, object]] | None = None
            sample_rows: list[dict[str, object]] = []
            for candidate_index in range(max(1, int(cfg.candidates_per_sample))):
                candidate_seed = int(cfg.seed + sample_index * 1777 + candidate_index * 131)
                morphology_mask, plan = build_morphology_calibrated_mask(
                    prior,
                    sample.cls_name,
                    local_bbox,
                    target_key=target_key,
                    target_size=(cfg.resolution, cfg.resolution),
                    target_source_size=expanded_crop.size,
                    candidate_index=candidate_index,
                    seed=candidate_seed,
                    feather_radius=cfg.feather_radius,
                )
                masks = _build_lgi_masks(morphology_mask)
                masks.core.save(mask_dir / f"{target_key}_c{candidate_index:02d}_M_core.png")
                masks.shell.save(mask_dir / f"{target_key}_c{candidate_index:02d}_M_shell.png")
                masks.out.save(mask_dir / f"{target_key}_c{candidate_index:02d}_M_out.png")
                masks.edit.save(mask_dir / f"{target_key}_c{candidate_index:02d}_M_edit.png")
                pseudo_normal_crop = _build_pseudo_normal_canvas(working_crop, masks)
                pseudo_path = pseudo_dir / f"{target_key}_c{candidate_index:02d}.png"
                pseudo_normal_crop.save(pseudo_path)

                generator = torch.Generator(device=device).manual_seed(candidate_seed)
                context_latents = self._encode_image(pipe, pseudo_normal_crop, cfg.resolution, dtype)
                latent_h, latent_w = context_latents.shape[-2:]
                latent_core = mask_to_latent_tensor(masks.core, (latent_h, latent_w), device, context_latents.dtype)
                latent_shell = mask_to_latent_tensor(masks.shell, (latent_h, latent_w), device, context_latents.dtype)
                latent_out = mask_to_latent_tensor(masks.out, (latent_h, latent_w), device, context_latents.dtype)
                latent_core, latent_shell, latent_out = normalize_latent_triplet(latent_core, latent_shell, latent_out)
                context_noise = torch.randn(
                    context_latents.shape,
                    device=device,
                    dtype=context_latents.dtype,
                    generator=generator,
                )
                latents = torch.randn(
                    context_latents.shape,
                    device=device,
                    dtype=context_latents.dtype,
                    generator=generator,
                )
                eta_values: list[float] = []
                with torch.no_grad():
                    timesteps = pipe.scheduler.timesteps
                    for step_idx, timestep in enumerate(timesteps):
                        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
                        noise_pred = pipe.unet(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=prompt_embeds,
                        ).sample
                        if do_cfg:
                            noise_uncond, noise_text = noise_pred.chunk(2)
                            noise_pred = noise_uncond + cfg.guidance_scale * (noise_text - noise_uncond)
                        refined = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
                        if step_idx + 1 < len(timesteps):
                            next_t = timesteps[step_idx + 1].reshape(1).to(device=device)
                            context_for_next = self._add_noise(pipe.scheduler, context_latents, context_noise, next_t)
                        else:
                            context_for_next = context_latents
                        eta = transition_eta(
                            pipe.scheduler,
                            timestep.reshape(1).to(device=device),
                            eta_min=cfg.eta_min,
                            eta_max=cfg.eta_max,
                            step_idx=step_idx,
                            num_steps=len(timesteps),
                            device=device,
                            dtype=context_latents.dtype,
                        )
                        propagated_background = propagate_background_latents(
                            context_for_next,
                            latent_out,
                            sigma=cfg.background_propagation_sigma,
                        )
                        shell_eta = (eta * float(cfg.shell_strength)).clamp(0.0, 1.0)
                        shell_latents = (1.0 - shell_eta) * propagated_background + shell_eta * refined
                        core_strength = max(0.0, min(1.0, float(cfg.core_strength)))
                        core_latents = (1.0 - core_strength) * context_for_next + core_strength * refined
                        latents = latent_out * context_for_next + latent_shell * shell_latents + latent_core * core_latents
                        eta_values.append(float(eta.detach().cpu().flatten()[0]))

                generated_crop = self._decode_latents(pipe, latents).resize(expanded_crop.size, Image.Resampling.LANCZOS)
                generated_crop = _neutralize_chroma_if_monochrome(
                    generated_crop,
                    expanded_crop,
                    force=str(domain).lower() == "steel",
                )
                gen_crop_path = gen_crop_dir / f"{target_key}_c{candidate_index:02d}.png"
                generated_crop.save(gen_crop_path)
                crop_mask = masks.edit.resize(expanded_crop.size, Image.Resampling.BILINEAR).filter(
                    ImageFilter.GaussianBlur(radius=1.0)
                )
                expanded_canvas = original.copy()
                expanded_canvas.paste(generated_crop, expanded[:2])
                full_mask = Image.new("L", original.size, 0)
                full_mask.paste(crop_mask, expanded[:2])
                final = create_composite_preserving_background(original, expanded_canvas, full_mask)
                candidate_path = candidate_dir / f"{target_key}_c{candidate_index:02d}.png"
                final.save(candidate_path)

                generated_working_crop = generated_crop.resize(working_crop.size, Image.Resampling.LANCZOS)
                quality = _quality_domains(working_crop, generated_working_crop, masks)
                selection_score = float(quality["total"])
                row: dict[str, object] = {
                    "target_key": target_key,
                    "source_key": sample.source_stem,
                    "object_index": int(getattr(sample, "object_index", 0)),
                    "object_count": len(getattr(sample, "objects", ()) or ()) or 1,
                    "class": sample.cls_name,
                    "prompt": prompt,
                    "candidate_index": int(candidate_index),
                    "candidate_path": str(candidate_path.resolve()),
                    "selected": False,
                    "method": "context_lgi",
                    "paper_reference": "Context-LGI: context-conditioned latent generative inpainting",
                    "spatial_code": {
                        "source_bbox": [int(v) for v in sample.bbox],
                        "expanded_bbox": [int(v) for v in expanded],
                        "defect_bbox_in_expanded": [int(v) for v in local_bbox],
                        "scaled_bbox": [int(v) for v in scaled_local_bbox],
                    },
                    "morphology_code": asdict(plan),
                    "latent_inpainting": {
                        "form": "I_syn = Composite(I_real, G(z_space,z_shape,z_texture,C), M)",
                        "resolution": int(cfg.resolution),
                        "steps": int(cfg.num_inference_steps),
                        "eta_min": float(cfg.eta_min),
                        "eta_max": float(cfg.eta_max),
                        "shell_strength": float(cfg.shell_strength),
                        "core_strength": float(cfg.core_strength),
                        "background_propagation_sigma": float(cfg.background_propagation_sigma),
                        "eta_mean": float(np.mean(eta_values)) if eta_values else None,
                    },
                    "mask_stats": masks.stats,
                    "quality": quality,
                    "selection_score": selection_score,
                    "paths": {
                        "candidate": str(candidate_path.resolve()),
                        "pseudo_normal_crop": str(pseudo_path.resolve()),
                        "generated_crop": str(gen_crop_path.resolve()),
                        "core_mask": str((mask_dir / f"{target_key}_c{candidate_index:02d}_M_core.png").resolve()),
                        "shell_mask": str((mask_dir / f"{target_key}_c{candidate_index:02d}_M_shell.png").resolve()),
                        "out_mask": str((mask_dir / f"{target_key}_c{candidate_index:02d}_M_out.png").resolve()),
                        "edit_mask": str((mask_dir / f"{target_key}_c{candidate_index:02d}_M_edit.png").resolve()),
                    },
                    "label_contract": "source-label-inherited",
                }
                sample_rows.append(row)
                if best is None or selection_score > best[0]:
                    best = (selection_score, final, masks.edit, row)

            if best is None:
                continue
            _, selected_image, selected_mask, selected_row = best
            selected_row["selected"] = True
            out_path = image_dir / f"{target_key}.png"
            selected_image.save(out_path)
            selected_mask.save(mask_dir / f"{target_key}_selected_M_edit.png")
            selected_row["output_path"] = str(out_path.resolve())
            target_selected = (
                selected_row.get("utility_causal_spectrum", {}).get("target_spectrum")
                if isinstance(selected_row.get("utility_causal_spectrum"), dict)
                else None
            )
            if isinstance(target_selected, dict):
                selected_spectra_by_class.setdefault(sample.cls_name, []).append(dict(target_selected))
                orientation_bin = int(float(target_selected.get("orientation_bin", 0)))
                counts = orientation_counts_by_class.setdefault(sample.cls_name, {})
                counts[orientation_bin] = int(counts.get(orientation_bin, 0)) + 1
            rows.extend(sample_rows)
            outputs.append(out_path)
            if cfg.use_ucs:
                selected_spectrum = (
                    selected_row.get("utility_causal_spectrum", {})
                    if isinstance(selected_row.get("utility_causal_spectrum"), dict)
                    else {}
                )
                target_selected = selected_spectrum.get("target_spectrum", {}) if isinstance(selected_spectrum, dict) else {}
                if isinstance(target_selected, dict):
                    selected_spectra_by_class.setdefault(sample.cls_name, []).append(dict(target_selected))
                    orientation_bin = int(float(target_selected.get("orientation_bin", 0) or 0))
                    counts = orientation_counts_by_class.setdefault(sample.cls_name, {})
                    counts[orientation_bin] = int(counts.get(orientation_bin, 0)) + 1

        score_path = artifacts_dir / "candidate_scores.jsonl"
        score_mode = "a" if score_path.exists() else "w"
        with score_path.open(score_mode, encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "method": "context_lgi",
            "model_dir": self.model_dir,
            "generated_images": len(outputs),
            "image_dir": str(image_dir.resolve()),
            "score_path": str(score_path.resolve()),
            "prompts_by_class": dict(sorted(prompts_by_class.items())),
            "config": asdict(cfg),
            "innovation": {
                "context_coordinate_generation": "space, shape and texture variables are sampled as latent inpainting coordinates",
                "context_preserved_latent_inpainting": "background latent is injected at every denoising step through M_out while M_core remains generative",
            },
        }
        (output_dir / "context_lgi_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return outputs


class DELGIGenerator(ReferenceCMDPGenerator):
    """Defect-energy projected latent generative inpainting."""

    def _build_drr_donor_residual(
        self,
        pipe,
        donor: DefectSample,
        *,
        prior: MorphologyPrior,
        config: DELGIConfig,
        device: torch.device,
        dtype: torch.dtype,
        donor_dir: Path,
        cache: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, object]]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        donor_key = f"{_sample_target_key(donor)}_o{int(getattr(donor, 'object_index', 0)):02d}"
        cache_key = f"{donor_key}_r{int(config.resolution)}_d{float(config.dilation_factor):.3f}_s{float(config.pseudo_suppression_strength):.3f}"
        cached = cache.get(cache_key)
        if cached is not None:
            delta, mask, info = cached
            return delta.to(device=device, dtype=dtype), mask.to(device=device, dtype=dtype), dict(info)

        original = Image.open(donor.image_path).convert("RGB")
        expanded = expand_bbox(donor.bbox, original.size, config.dilation_factor)
        expanded_crop = original.crop(expanded)
        local_bbox = bbox_inside_crop(donor.bbox, expanded)
        working_crop = expanded_crop.resize((config.resolution, config.resolution), Image.Resampling.LANCZOS)
        donor_seed = int(config.seed + _stable_int_from_text(donor_key)) & 0xFFFFFFFF
        morphology_mask, _plan = build_morphology_calibrated_mask(
            prior,
            donor.cls_name,
            local_bbox,
            target_key=donor_key,
            target_size=(config.resolution, config.resolution),
            target_source_size=expanded_crop.size,
            candidate_index=0,
            seed=donor_seed,
            feather_radius=config.feather_radius,
        )
        donor_masks = _build_lgi_masks(morphology_mask)
        donor_pseudo = _build_pseudo_normal_canvas(
            working_crop,
            donor_masks,
            suppression_strength=max(0.35, float(config.pseudo_suppression_strength)),
        )
        donor_dir.mkdir(parents=True, exist_ok=True)
        crop_path = donor_dir / f"{donor_key}_crop.png"
        pseudo_path = donor_dir / f"{donor_key}_pseudo.png"
        mask_path = donor_dir / f"{donor_key}_M_edit.png"
        if not crop_path.exists():
            working_crop.save(crop_path)
        if not pseudo_path.exists():
            donor_pseudo.save(pseudo_path)
        if not mask_path.exists():
            donor_masks.edit.save(mask_path)

        defect_latents = self._encode_image(pipe, working_crop, config.resolution, dtype)
        pseudo_latents = self._encode_image(pipe, donor_pseudo, config.resolution, dtype)
        latent_h, latent_w = defect_latents.shape[-2:]
        donor_core = mask_to_latent_tensor(donor_masks.core, (latent_h, latent_w), device, dtype)
        donor_delta = (defect_latents - pseudo_latents).detach()
        info: dict[str, object] = {
            "donor_source_key": donor.source_stem,
            "donor_target_key": _sample_target_key(donor),
            "donor_object_index": int(getattr(donor, "object_index", 0)),
            "donor_class": donor.cls_name,
            "donor_crop": str(crop_path.resolve()),
            "donor_pseudo_normal_crop": str(pseudo_path.resolve()),
            "donor_edit_mask": str(mask_path.resolve()),
        }
        cache[cache_key] = (donor_delta.detach().cpu(), donor_core.detach().cpu(), dict(info))
        return donor_delta, donor_core, info

    def generate_de_lgi(
        self,
        samples: Sequence[DefectSample],
        output_dir: Path,
        *,
        domain: str,
        reference_samples: Optional[Sequence[DefectSample]] = None,
        config: DELGIConfig | None = None,
    ) -> list[Path]:
        cfg = config or DELGIConfig()
        pipe = self._load_pipe()
        device = pipe.unet.device
        dtype = self._dtype()
        output_dir = Path(output_dir)
        image_dir = output_dir / "images"
        artifacts_dir = output_dir / "artifacts" / "de_lgi"
        candidate_dir = artifacts_dir / "candidates"
        crop_dir = artifacts_dir / "expanded_crops"
        pseudo_dir = artifacts_dir / "pseudo_normal_crops"
        donor_dir = artifacts_dir / "residual_donors"
        gen_crop_dir = artifacts_dir / "generated_expanded_crops"
        mask_dir = artifacts_dir / "masks"
        structure_dir = artifacts_dir / "srt_structure_fields"
        for path in (image_dir, artifacts_dir, candidate_dir, crop_dir, pseudo_dir, donor_dir, gen_crop_dir, mask_dir, structure_dir):
            path.mkdir(parents=True, exist_ok=True)

        selected_samples = list(samples)
        references = list(reference_samples or selected_samples)
        reference_bank_by_class: dict[str, list[DefectSample]] = {}
        for ref in references:
            reference_bank_by_class.setdefault(ref.cls_name, []).append(ref)
        donor_latent_cache: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, object]]] = {}
        drr_enabled = bool(cfg.use_ucs and cfg.use_drr and float(cfg.residual_bank_mix) > 0)
        srt_enabled = bool(cfg.use_ucs and cfg.use_srt)
        prior = MorphologyPrior.from_samples(references)
        prior.write_manifest(artifacts_dir / "morphology_prior.json")
        energy_prior = DefectEnergyPrior.from_samples(
            references,
            resolution=int(cfg.resolution),
            dilation_factor=float(cfg.dilation_factor),
            latent_energy_scale=float(cfg.latent_energy_scale),
            latent_energy_floor=float(cfg.latent_energy_floor),
            latent_energy_ceiling=float(cfg.latent_energy_ceiling),
        )
        (artifacts_dir / "energy_prior.json").write_text(
            json.dumps(energy_prior.to_manifest(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        spectrum_prior = DefectSpectrumPrior.from_samples(
            references,
            resolution=int(cfg.resolution),
            dilation_factor=float(cfg.dilation_factor),
        )
        (artifacts_dir / "defect_spectrum_prior.json").write_text(
            json.dumps(spectrum_prior.to_manifest(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        pipe.scheduler.set_timesteps(int(cfg.num_inference_steps), device=device)
        do_cfg = cfg.guidance_scale > 1.0
        prompt_cache: dict[str, torch.Tensor] = {}
        prompts_by_class: dict[str, str] = {}
        rows: list[dict[str, object]] = []
        trace_rows: list[dict[str, object]] = []
        outputs: list[Path] = []
        selected_spectra_by_class: dict[str, list[dict[str, object]]] = {}
        orientation_counts_by_class: dict[str, dict[int, int]] = {}

        for sample_index, sample in enumerate(track(selected_samples, description="DE-LGI latent projection")):
            target_key = _sample_target_key(sample)
            prompt = reference_caption(domain, sample.cls_name)
            prompts_by_class.setdefault(sample.cls_name, prompt)
            prompt_embeds = prompt_cache.get(prompt)
            if prompt_embeds is None:
                prompt_embeds = self._encode_prompt(pipe, prompt, cfg.guidance_scale)
                prompt_cache[prompt] = prompt_embeds

            original = Image.open(sample.image_path).convert("RGB")
            expanded = expand_bbox(sample.bbox, original.size, cfg.dilation_factor)
            expanded_crop = original.crop(expanded)
            local_bbox = bbox_inside_crop(sample.bbox, expanded)
            working_crop = expanded_crop.resize((cfg.resolution, cfg.resolution), Image.Resampling.LANCZOS)
            scaled_local_bbox = scale_bbox_to_size(local_bbox, expanded_crop.size, working_crop.size)
            crop_dir.joinpath(f"{target_key}.png").parent.mkdir(parents=True, exist_ok=True)
            working_crop.save(crop_dir / f"{target_key}.png")
            try:
                source_spectrum = _spectrum_observation_to_dict(
                    _estimate_sample_spectrum(
                        sample,
                        resolution=int(cfg.resolution),
                        dilation_factor=float(cfg.dilation_factor),
                    )
                )
            except Exception:
                source_spectrum = _spectrum_observation_to_dict(_fallback_spectrum_observation(sample.cls_name))

            best: tuple[float, Image.Image, Image.Image, dict[str, object]] | None = None
            sample_rows: list[dict[str, object]] = []
            for candidate_index in range(max(1, int(cfg.candidates_per_sample))):
                candidate_seed = int(cfg.seed + sample_index * 1777 + candidate_index * 131)
                target_energy = energy_prior.target_energy(
                    sample.cls_name,
                    seed=candidate_seed,
                    jitter=float(cfg.energy_jitter),
                )
                energy_profile = energy_prior.profile(sample.cls_name)
                target_spectrum = _sample_distant_target_spectrum(
                    spectrum_prior,
                    sample.cls_name,
                    source_spectrum=source_spectrum if (drr_enabled or srt_enabled or bool(cfg.use_hdsi)) else None,
                    seed=candidate_seed,
                    jitter=float(cfg.spectrum_jitter),
                    utility_guidance=float(cfg.utility_guidance),
                    coverage_counts=orientation_counts_by_class.get(sample.cls_name),
                    history=selected_spectra_by_class.get(sample.cls_name, []),
                    config=cfg,
                )
                if cfg.use_ucs:
                    target_energy = float(target_energy) * float(target_spectrum.get("energy_multiplier", 1.0))
                spectrum_profile = spectrum_prior.profile(sample.cls_name)
                diversity_gain = _spectrum_diversity_gain(
                    {str(k): v for k, v in target_spectrum.items()},
                    selected_spectra_by_class.get(sample.cls_name, []),
                )
                morphology_mask, plan = build_morphology_calibrated_mask(
                    prior,
                    sample.cls_name,
                    local_bbox,
                    target_key=target_key,
                    target_size=(cfg.resolution, cfg.resolution),
                    target_source_size=expanded_crop.size,
                    candidate_index=candidate_index,
                    seed=candidate_seed,
                    feather_radius=cfg.feather_radius,
                )
                srt_plan: SRTStructurePlan | None = None
                source_structure_mask: Image.Image | None = None
                structure_mask = morphology_mask
                if srt_enabled:
                    structure_mask, source_structure_mask, srt_plan = _build_srt_structure_field(
                        working_crop,
                        scaled_local_bbox,
                        morphology_mask,
                        cls_name=sample.cls_name,
                        target_key=target_key,
                        target_spectrum=target_spectrum,
                        seed=candidate_seed + 104729,
                        config=cfg,
                    )
                    structure_mask.save(structure_dir / f"{target_key}_c{candidate_index:02d}_phi.png")
                    source_structure_mask.save(structure_dir / f"{target_key}_c{candidate_index:02d}_phi_source.png")
                masks = _build_lgi_masks(structure_mask)
                canvas_mask = _union_mask_images(masks.edit, source_structure_mask) if source_structure_mask is not None else masks.edit
                canvas_masks = _build_lgi_masks(canvas_mask)
                masks.core.save(mask_dir / f"{target_key}_c{candidate_index:02d}_M_core.png")
                masks.shell.save(mask_dir / f"{target_key}_c{candidate_index:02d}_M_shell.png")
                masks.out.save(mask_dir / f"{target_key}_c{candidate_index:02d}_M_out.png")
                masks.edit.save(mask_dir / f"{target_key}_c{candidate_index:02d}_M_edit.png")
                if srt_enabled:
                    canvas_masks.edit.save(mask_dir / f"{target_key}_c{candidate_index:02d}_M_canvas_edit.png")
                pseudo_normal_crop = _build_pseudo_normal_canvas(
                    working_crop,
                    canvas_masks,
                    suppression_strength=float(cfg.pseudo_suppression_strength) if (drr_enabled or srt_enabled) else 0.0,
                )
                pseudo_path = pseudo_dir / f"{target_key}_c{candidate_index:02d}.png"
                pseudo_normal_crop.save(pseudo_path)

                generator = torch.Generator(device=device).manual_seed(candidate_seed)
                context_latents = self._encode_image(pipe, pseudo_normal_crop, cfg.resolution, dtype)
                latent_h, latent_w = context_latents.shape[-2:]
                latent_core = mask_to_latent_tensor(masks.core, (latent_h, latent_w), device, context_latents.dtype)
                latent_shell = mask_to_latent_tensor(masks.shell, (latent_h, latent_w), device, context_latents.dtype)
                latent_out = mask_to_latent_tensor(masks.out, (latent_h, latent_w), device, context_latents.dtype)
                latent_core, latent_shell, latent_out = normalize_latent_triplet(latent_core, latent_shell, latent_out)
                srt_delta: torch.Tensor | None = None
                hdsi_pd_delta: torch.Tensor | None = None
                srt_info: dict[str, object] = {}
                residual_recomposition_score = 0.0
                if srt_enabled and source_structure_mask is not None:
                    original_latents = self._encode_image(pipe, working_crop, cfg.resolution, dtype)
                    latent_source_core = mask_to_latent_tensor(
                        source_structure_mask,
                        (latent_h, latent_w),
                        device,
                        context_latents.dtype,
                    )
                    source_delta = (original_latents - context_latents).detach()
                    srt_delta, srt_transport_info = _transport_srt_residual_to_structure(
                        source_delta,
                        latent_source_core,
                        latent_core,
                        latent_shell,
                        target_energy=float(target_energy),
                        target_spectrum=target_spectrum,
                        config=cfg,
                    )
                    srt_info = {
                        "mode": "structure_residual_transport",
                        "formula": (
                            "z_g = E(phi', evidence, background) * [z_normal + G(phi') * T(z_defect-z_normal | phi_source -> phi')]"
                            if bool(cfg.use_e_srt)
                            else "z_g = z_normal + G(phi') * T(z_defect-z_normal | phi_source -> phi')"
                        ),
                        "bbox_label_policy": (
                            "structure-evidence-consistent"
                            if bool(cfg.use_seca)
                            else "structure-field-derived"
                            if bool(cfg.srt_bbox_update)
                            else "source-label-inherited"
                        ),
                        "structure_field": asdict(srt_plan) if srt_plan is not None else {},
                        **srt_transport_info,
                    }
                    donor_count = max(0, min(1, int(cfg.band_donor_count) - 1))
                    if donor_count > 0:
                        raw_donors = _select_residual_donors(
                            reference_bank_by_class,
                            sample,
                            seed=candidate_seed + 9311,
                            count=max(donor_count + 2, 3),
                        )
                        donors: list[DefectSample] = []
                        seen_donor_keys: set[str] = set()
                        for donor in raw_donors:
                            donor_key = f"{_sample_target_key(donor)}_o{int(getattr(donor, 'object_index', 0)):02d}"
                            if donor_key in seen_donor_keys:
                                continue
                            if donor.source_stem == sample.source_stem and len(raw_donors) > donor_count:
                                continue
                            donors.append(donor)
                            seen_donor_keys.add(donor_key)
                            if len(donors) >= donor_count:
                                break
                        if len(donors) < donor_count:
                            for donor in raw_donors:
                                if donor in donors:
                                    continue
                                donors.append(donor)
                                if len(donors) >= donor_count:
                                    break
                        srt_residuals = [srt_delta]
                        donor_records: list[dict[str, object]] = []
                        same_source_flags: list[bool] = []
                        band_roles = ("source_low_frequency_shape", "donor_mid_frequency_texture", "donor_high_frequency_edges")
                        for donor_idx, donor in enumerate(donors):
                            donor_delta, donor_mask, donor_info = self._build_drr_donor_residual(
                                pipe,
                                donor,
                                prior=prior,
                                config=cfg,
                                device=device,
                                dtype=context_latents.dtype,
                                donor_dir=donor_dir,
                                cache=donor_latent_cache,
                            )
                            donor_transport, donor_transport_info = _transport_srt_residual_to_structure(
                                donor_delta,
                                donor_mask,
                                latent_core,
                                latent_shell,
                                target_energy=float(target_energy),
                                target_spectrum=target_spectrum,
                                config=cfg,
                            )
                            srt_residuals.append(donor_transport)
                            same_source = donor.source_stem == sample.source_stem
                            same_source_flags.append(bool(same_source))
                            donor_records.append({
                                **donor_info,
                                **donor_transport_info,
                                "band_role": band_roles[min(donor_idx + 1, len(band_roles) - 1)],
                                "band_index": int(donor_idx + 1),
                                "same_source_as_target": bool(same_source),
                            })
                        if len(srt_residuals) > 1:
                            recomposed_delta, band_info = _compose_bandwise_drr_residual(
                                srt_residuals,
                                latent_core,
                                target_spectrum=target_spectrum,
                                seed=candidate_seed + 6151,
                                config=cfg,
                            )
                            source_anchor = float(np.clip(cfg.srt_source_preservation, 0.0, 0.45))
                            srt_delta = source_anchor * srt_delta + (1.0 - source_anchor) * recomposed_delta
                            distinct_sources = len({str(record.get("donor_source_key", "")) for record in donor_records})
                            non_target_ratio = 1.0 - (sum(1 for flag in same_source_flags if flag) / max(1, len(same_source_flags)))
                            residual_recomposition_score = float(
                                np.clip(
                                    0.40
                                    + 0.25 * non_target_ratio
                                    + 0.20 * min(1.0, distinct_sources / max(1, donor_count))
                                    + 0.15 * float(np.clip(cfg.band_recomposition_strength, 0.0, 1.0)),
                                    0.0,
                                    1.0,
                                )
                            )
                            srt_info["srt_residual_recomposition"] = {
                                "mode": "multi_source_band_recomposition",
                                "source_anchor": float(source_anchor),
                                "band_donors": donor_records,
                                "residual_recomposition_score": float(residual_recomposition_score),
                                **band_info,
                            }
                    if bool(cfg.use_hdsi_pd) and bool(cfg.use_hdsi):
                        prototype_count = max(1, int(cfg.hdsi_pd_prototype_count))
                        prototype_donors, prototype_records = _select_hdsi_pd_prototype_donors(
                            reference_bank_by_class,
                            sample,
                            target_spectrum=target_spectrum,
                            source_spectrum=source_spectrum,
                            spectrum_prior=spectrum_prior,
                            seed=candidate_seed + 15485863,
                            count=prototype_count,
                            config=cfg,
                        )
                        prototype_deltas: list[torch.Tensor] = []
                        enriched_records: list[dict[str, object]] = []
                        for donor_idx, donor in enumerate(prototype_donors):
                            donor_delta, donor_mask, donor_info = self._build_drr_donor_residual(
                                pipe,
                                donor,
                                prior=prior,
                                config=cfg,
                                device=device,
                                dtype=context_latents.dtype,
                                donor_dir=donor_dir,
                                cache=donor_latent_cache,
                            )
                            donor_transport, donor_transport_info = _transport_srt_residual_to_structure(
                                donor_delta,
                                donor_mask,
                                latent_core,
                                latent_shell,
                                target_energy=float(target_energy),
                                target_spectrum=target_spectrum,
                                config=cfg,
                            )
                            prototype_deltas.append(donor_transport)
                            prototype_meta = prototype_records[min(donor_idx, max(0, len(prototype_records) - 1))] if prototype_records else {}
                            enriched_records.append(
                                {
                                    **prototype_meta,
                                    **donor_info,
                                    **donor_transport_info,
                                    "prototype_index": int(donor_idx),
                                    "same_source_as_target": bool(donor.source_stem == sample.source_stem),
                                }
                            )
                        hdsi_pd_delta, hdsi_pd_info = _compose_hdsi_pd_prototype_delta(
                            prototype_deltas,
                            latent_core,
                            target_energy=float(target_energy),
                            target_spectrum=target_spectrum,
                            seed=candidate_seed + 32452843,
                            config=cfg,
                        )
                        if hdsi_pd_delta is not None:
                            hdsi_score = float(
                                np.clip(
                                    float(target_spectrum.get("hdsi_intervention_score", 0.0))
                                    * float(target_spectrum.get("hdsi_validity_score", 1.0)),
                                    0.0,
                                    1.0,
                                )
                            )
                            preset_key = str(cfg.method_preset).lower().replace("_", "-")
                            if preset_key == "e-srt-hdsi-pd":
                                prototype_mix = float(np.clip(cfg.hdsi_pd_strength, 0.0, 1.0)) * float(0.58 + 0.42 * hdsi_score)
                                prototype_mix = float(np.clip(prototype_mix, 0.34, 0.78))
                            else:
                                prototype_mix = float(np.clip(cfg.hdsi_pd_strength, 0.0, 1.0)) * float(0.46 + 0.54 * hdsi_score)
                            srt_delta = (1.0 - prototype_mix) * srt_delta + prototype_mix * hdsi_pd_delta
                            distinct_sources = len({str(record.get("donor_source_key", "")) for record in enriched_records})
                            residual_recomposition_score = max(
                                float(residual_recomposition_score),
                                float(np.clip(0.56 + 0.18 * hdsi_score + 0.16 * min(1.0, distinct_sources / max(1, prototype_count)), 0.0, 1.0)),
                            )
                            srt_info["hdsi_phase_detail_intervention"] = {
                                "definition": "HDSI-PD injects hard-spectrum real defect prototype phase/detail residuals into phi' while preserving the source bbox contract.",
                                "prototype_mix": float(prototype_mix),
                                "prototype_selection": enriched_records,
                                "residual_recomposition_score": float(residual_recomposition_score),
                                **hdsi_pd_info,
                            }
                        else:
                            srt_info["hdsi_phase_detail_intervention"] = {
                                "definition": "HDSI-PD requested but no valid prototype residual was available.",
                                "prototype_selection": enriched_records,
                                **hdsi_pd_info,
                            }
                context_noise = torch.randn(
                    context_latents.shape,
                    device=device,
                    dtype=context_latents.dtype,
                    generator=generator,
                )
                latents = torch.randn(
                    context_latents.shape,
                    device=device,
                    dtype=context_latents.dtype,
                    generator=generator,
                )
                drr_donor_delta: torch.Tensor | None = None
                drr_info: dict[str, object] = {}
                if drr_enabled:
                    donor_count = max(1, int(cfg.band_donor_count)) if cfg.use_band_recomposition else 1
                    donors = _select_residual_donors(
                        reference_bank_by_class,
                        sample,
                        seed=candidate_seed + 7919,
                        count=donor_count,
                    )
                    adapted_residuals: list[torch.Tensor] = []
                    donor_records: list[dict[str, object]] = []
                    same_source_flags: list[bool] = []
                    band_roles = ("low_frequency_shape", "mid_frequency_texture", "high_frequency_edges")
                    for donor_idx, donor in enumerate(donors):
                        donor_delta, donor_mask, donor_info = self._build_drr_donor_residual(
                            pipe,
                            donor,
                            prior=prior,
                            config=cfg,
                            device=device,
                            dtype=context_latents.dtype,
                            donor_dir=donor_dir,
                            cache=donor_latent_cache,
                        )
                        adapted_delta, adapt_info = _adapt_drr_residual_to_target(
                            donor_delta,
                            donor_mask,
                            latent_core,
                            latent_shell,
                            target_spectrum=target_spectrum,
                            seed=candidate_seed + 3571 + donor_idx * 431,
                            config=cfg,
                        )
                        same_source = donor.source_stem == sample.source_stem
                        same_source_flags.append(bool(same_source))
                        adapted_residuals.append(adapted_delta)
                        donor_records.append({
                            **donor_info,
                            **adapt_info,
                            "band_role": band_roles[min(donor_idx, len(band_roles) - 1)],
                            "band_index": int(donor_idx),
                            "same_source_as_target": bool(same_source),
                        })
                    if adapted_residuals:
                        if cfg.use_band_recomposition and len(adapted_residuals) > 1:
                            drr_donor_delta, band_info = _compose_bandwise_drr_residual(
                                adapted_residuals,
                                latent_core,
                                target_spectrum=target_spectrum,
                                seed=candidate_seed + 6151,
                                config=cfg,
                            )
                            drr_info = {
                                "mode": "band_wise_residual_recomposition",
                                "band_donors": donor_records,
                                **band_info,
                                "residual_bank_mix": float(cfg.residual_bank_mix),
                            }
                        else:
                            drr_donor_delta = adapted_residuals[0]
                            drr_info = {
                                "mode": "single_donor_residual_recomposition",
                                **donor_records[0],
                                "residual_bank_mix": float(cfg.residual_bank_mix),
                            }
                        distinct_sources = len({str(record.get("donor_source_key", "")) for record in donor_records})
                        non_target_ratio = 1.0 - (sum(1 for flag in same_source_flags if flag) / max(1, len(same_source_flags)))
                        residual_recomposition_score = float(np.clip(0.55 + 0.30 * non_target_ratio + 0.15 * min(1.0, distinct_sources / 3.0), 0.0, 1.0))
                eta_values: list[float] = []
                candidate_traces: list[dict[str, object]] = []
                last_trace: dict[str, float] = {}
                with torch.no_grad():
                    timesteps = pipe.scheduler.timesteps
                    for step_idx, timestep in enumerate(timesteps):
                        step_fraction = float(step_idx / max(1, len(timesteps) - 1))
                        srt_pre_trace: dict[str, float] = {}
                        if srt_delta is not None:
                            context_for_step = self._add_noise(
                                pipe.scheduler,
                                context_latents,
                                context_noise,
                                timestep.reshape(1).to(device=device),
                            )
                            latents, srt_pre_trace = _apply_srt_structure_regeneration(
                                latents,
                                context_for_step,
                                srt_delta.to(device=latents.device, dtype=latents.dtype),
                                latent_core,
                                target_energy=float(target_energy),
                                step_fraction=step_fraction,
                                config=cfg,
                            )
                        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
                        noise_pred = pipe.unet(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=prompt_embeds,
                        ).sample
                        if do_cfg:
                            noise_uncond, noise_text = noise_pred.chunk(2)
                            noise_pred = noise_uncond + cfg.guidance_scale * (noise_text - noise_uncond)
                        refined = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
                        if step_idx + 1 < len(timesteps):
                            next_t = timesteps[step_idx + 1].reshape(1).to(device=device)
                            context_for_next = self._add_noise(pipe.scheduler, context_latents, context_noise, next_t)
                        else:
                            context_for_next = context_latents
                        eta = transition_eta(
                            pipe.scheduler,
                            timestep.reshape(1).to(device=device),
                            eta_min=cfg.eta_min,
                            eta_max=cfg.eta_max,
                            step_idx=step_idx,
                            num_steps=len(timesteps),
                            device=device,
                            dtype=context_latents.dtype,
                        )
                        propagated_background = propagate_background_latents(
                            context_for_next,
                            latent_out,
                            sigma=cfg.background_propagation_sigma,
                        )
                        delta_raw = refined - propagated_background
                        srt_step_trace: dict[str, float] = {}
                        drr_step_trace: dict[str, float] = {}
                        if srt_delta is not None:
                            delta_raw, srt_step_trace = _mix_srt_residual(
                                delta_raw,
                                srt_delta.to(device=delta_raw.device, dtype=delta_raw.dtype),
                                latent_core,
                                target_energy=float(target_energy),
                                step_fraction=step_fraction,
                                config=cfg,
                            )
                        if drr_donor_delta is not None:
                            delta_raw, drr_step_trace = _mix_drr_residual(
                                delta_raw,
                                drr_donor_delta.to(device=delta_raw.device, dtype=delta_raw.dtype),
                                latent_core,
                                target_energy=float(target_energy),
                                step_fraction=step_fraction,
                                config=cfg,
                            )
                        if cfg.use_ucs and float(cfg.core_renoise_strength) > 0:
                            release = float(cfg.core_renoise_strength) * max(0.0, (0.42 - step_fraction) / 0.42)
                            if release > 0:
                                local_noise = torch.randn(
                                    delta_raw.shape,
                                    device=device,
                                    dtype=delta_raw.dtype,
                                    generator=generator,
                                )
                                delta_raw = delta_raw + latent_core * local_noise * float(target_energy) * release
                        if cfg.use_ucs:
                            delta_projected, trace = _project_causal_defect_spectrum(
                                delta_raw,
                                latent_core,
                                latent_shell,
                                latent_out,
                                eta=eta,
                                target_energy=target_energy,
                                target_spectrum=target_spectrum,
                                step_fraction=step_fraction,
                                config=cfg,
                            )
                        else:
                            delta_projected, trace = _project_defect_energy(
                                delta_raw,
                                latent_core,
                                latent_shell,
                                latent_out,
                                eta=eta,
                                target_energy=target_energy,
                                config=cfg,
                            )
                        hdsi_pd_step_trace: dict[str, float] = {}
                        if hdsi_pd_delta is not None:
                            detail_noise = (
                                torch.randn(
                                    delta_projected.shape,
                                    device=device,
                                    dtype=delta_projected.dtype,
                                    generator=generator,
                                )
                                if float(cfg.hdsi_pd_late_renoise_strength) > 0
                                else None
                            )
                            delta_projected, hdsi_pd_step_trace = _apply_hdsi_pd_detail_refinement(
                                delta_projected,
                                hdsi_pd_delta.to(device=delta_projected.device, dtype=delta_projected.dtype),
                                latent_core,
                                latent_shell,
                                target_energy=float(target_energy),
                                target_spectrum=target_spectrum,
                                step_fraction=step_fraction,
                                config=cfg,
                                detail_noise=detail_noise,
                            )
                        if drr_step_trace:
                            trace = {**trace, **drr_step_trace}
                        if srt_pre_trace:
                            trace = {**trace, **srt_pre_trace}
                        if srt_step_trace:
                            trace = {**trace, **srt_step_trace}
                        if hdsi_pd_step_trace:
                            trace = {**trace, **hdsi_pd_step_trace}
                        if srt_delta is not None and bool(cfg.use_e_srt):
                            delta_projected, e_srt_trace = _apply_e_srt_evidence_feedback(
                                delta_projected,
                                latent_core,
                                latent_shell,
                                latent_out,
                                target_energy=float(target_energy),
                                step_fraction=step_fraction,
                                config=cfg,
                            )
                            if e_srt_trace:
                                trace = {**trace, **e_srt_trace}
                        trace = {
                            **trace,
                            "projected_core_energy": float(
                                _masked_rms_tensor(_latent_highpass(delta_projected), latent_core).detach().cpu().item()
                            ),
                            "background_leakage": float(_masked_rms_tensor(delta_projected, latent_out).detach().cpu().item()),
                            "boundary_energy": float(
                                _masked_rms_tensor(_latent_highpass(delta_projected), latent_shell).detach().cpu().item()
                            ),
                        }
                        latents = context_for_next + delta_projected
                        eta_value = float(eta.detach().cpu().flatten()[0])
                        eta_values.append(eta_value)
                        last_trace = trace
                        if step_idx in {0, len(timesteps) // 2, len(timesteps) - 1}:
                            candidate_traces.append(
                                {
                                    "target_key": target_key,
                                    "class": sample.cls_name,
                                    "candidate_index": int(candidate_index),
                                    "step": int(step_idx),
                                    "eta": eta_value,
                                    **trace,
                                }
                            )

                generated_crop = self._decode_latents(pipe, latents).resize(expanded_crop.size, Image.Resampling.LANCZOS)
                generated_crop = _neutralize_chroma_if_monochrome(
                    generated_crop,
                    expanded_crop,
                    force=str(domain).lower() == "steel",
                )
                gen_crop_path = gen_crop_dir / f"{target_key}_c{candidate_index:02d}.png"
                generated_crop.save(gen_crop_path)
                crop_mask = (canvas_masks.edit if srt_enabled else masks.edit).resize(expanded_crop.size, Image.Resampling.BILINEAR).filter(
                    ImageFilter.GaussianBlur(radius=1.0)
                )
                expanded_canvas = original.copy()
                expanded_canvas.paste(generated_crop, expanded[:2])
                full_mask = Image.new("L", original.size, 0)
                full_mask.paste(crop_mask, expanded[:2])
                final = create_composite_preserving_background(original, expanded_canvas, full_mask)
                candidate_path = candidate_dir / f"{target_key}_c{candidate_index:02d}.png"
                final.save(candidate_path)

                generated_working_crop = generated_crop.resize(working_crop.size, Image.Resampling.LANCZOS)
                final_working_crop = final.crop(expanded).resize(working_crop.size, Image.Resampling.LANCZOS)
                target_quality = _quality_domains(working_crop, generated_working_crop, masks)
                s2i_metrics: dict[str, float | bool] = {}
                s2c_metrics: dict[str, object] = {}
                if srt_enabled:
                    canvas_quality = _quality_domains(working_crop, generated_working_crop, canvas_masks)
                    s2i_metrics = _structure_to_image_consistency(
                        working_crop,
                        generated_working_crop,
                        structure_mask,
                        canvas_masks.edit,
                        config=cfg,
                    )
                    srt_info["structure_to_image_consistency"] = s2i_metrics
                    if bool(cfg.use_hdsi_s2c):
                        s2c_metrics = _hdsi_s2c_consistency(
                            working_crop,
                            final_working_crop,
                            structure_mask,
                            canvas_masks.edit,
                            target_spectrum=target_spectrum,
                            srt_plan=srt_plan,
                            s2i_metrics=s2i_metrics,
                            config=cfg,
                        )
                        s2c_metrics["hdsi_s2c_measurement_stage"] = "final_composited_expanded_crop"
                        srt_info["spectrum_to_structure_consistency"] = s2c_metrics
                    base_quality = {
                        **target_quality,
                        "total": 0.62 * float(target_quality["total"]) + 0.38 * float(canvas_quality["total"]),
                        "outside_delta": float(canvas_quality["outside_delta"]),
                        "background_score": float(canvas_quality["background_score"]),
                        "canvas_total": float(canvas_quality["total"]),
                        "canvas_inside_delta": float(canvas_quality["inside_delta"]),
                        "canvas_boundary_delta": float(canvas_quality["boundary_delta"]),
                        "canvas_outside_delta": float(canvas_quality["outside_delta"]),
                        "canvas_background_score": float(canvas_quality["background_score"]),
                    }
                else:
                    base_quality = target_quality
                inherits_source_bbox = not (
                    srt_enabled
                    and srt_plan is not None
                    and (bool(cfg.srt_bbox_update) or bool(cfg.use_seca))
                )
                planned_generated_bbox_scaled = (
                    tuple(int(v) for v in srt_plan.generated_bbox_scaled)
                    if not inherits_source_bbox
                    else tuple(int(v) for v in scaled_local_bbox)
                )
                generated_bbox_scaled = planned_generated_bbox_scaled
                annotation_mask = canvas_masks.edit if srt_enabled else masks.edit
                seca_info: dict[str, object] = {}
                seca_mask_path: Path | None = None
                if srt_enabled and srt_plan is not None and bool(cfg.use_seca):
                    generated_bbox_scaled, seca_mask, seca_info = _derive_seca_annotation(
                        working_crop,
                        generated_working_crop,
                        planned_generated_bbox_scaled,
                        srt_plan.source_bbox_scaled,
                        structure_mask,
                        canvas_masks.edit,
                        cls_name=sample.cls_name,
                        config=cfg,
                    )
                    seca_mask_path = mask_dir / f"{target_key}_c{candidate_index:02d}_M_seca.png"
                    seca_mask.save(seca_mask_path)
                    annotation_mask = seca_mask if bool(seca_info.get("seca_pass", False)) else canvas_masks.edit
                    srt_info["structure_evidence_annotation"] = seca_info
                actual_energy = float(last_trace.get("projected_core_energy", 0.0))
                background_leakage = float(last_trace.get("background_leakage", 0.0))
                if cfg.use_ucs:
                    if srt_enabled:
                        if str(cfg.method_preset).lower().replace("_", "-") == "spec-srt":
                            quality = _spec_srt_quality(
                                base_quality,
                                cls_name=sample.cls_name,
                                energy_target=float(target_energy),
                                energy_actual=actual_energy,
                                background_leakage=background_leakage,
                                spectrum_match=float(last_trace.get("spectrum_match_score", 0.0)),
                                utility_score=float(target_spectrum.get("utility_score", 0.0)),
                                diversity_gain=diversity_gain,
                                structure_transport_score=float(srt_plan.structure_transport_score if srt_plan is not None else 0.0),
                                hdsi_intervention_score=float(target_spectrum.get("hdsi_intervention_score", 0.0)),
                                hdsi_validity_score=float(target_spectrum.get("hdsi_validity_score", 1.0)),
                                source_spectrum_distance=float(target_spectrum.get("source_spectrum_distance", 0.0)),
                                s2i_score=float(s2i_metrics.get("s2i_score", 0.0)),
                                s2i_visible_delta=float(s2i_metrics.get("s2i_visible_delta", 0.0)),
                                s2i_ratio=float(s2i_metrics.get("s2i_ratio", 0.0)),
                                s2i_gate_pass=bool(s2i_metrics.get("s2i_gate_pass", True)),
                                config=cfg,
                            )
                        else:
                            quality = _srt_de_lgi_quality(
                                base_quality,
                                energy_target=float(target_energy),
                                energy_actual=actual_energy,
                                background_leakage=background_leakage,
                                spectrum_match=float(last_trace.get("spectrum_match_score", 0.0)),
                                utility_score=float(target_spectrum.get("utility_score", 0.0)),
                                diversity_gain=diversity_gain,
                                structure_transport_score=float(srt_plan.structure_transport_score if srt_plan is not None else 0.0),
                                hdsi_intervention_score=float(target_spectrum.get("hdsi_intervention_score", 0.0)),
                                hdsi_validity_score=float(target_spectrum.get("hdsi_validity_score", 1.0)),
                                structure_novelty=float(1.0 - srt_plan.counterfactual_structure_iou if srt_plan is not None else 0.0),
                                residual_recomposition_score=float(residual_recomposition_score),
                                s2i_score=float(s2i_metrics.get("s2i_score", 0.0)),
                                s2i_visible_delta=float(s2i_metrics.get("s2i_visible_delta", 0.0)),
                                s2i_ratio=float(s2i_metrics.get("s2i_ratio", 0.0)),
                                s2i_gate_pass=bool(s2i_metrics.get("s2i_gate_pass", True)),
                                seca_score=float(seca_info.get("seca_score", 0.0)),
                                seca_gate_pass=bool(seca_info.get("seca_pass", True)),
                                config=cfg,
                            )
                        selection_score = float(quality["srt_de_lgi_total"])
                        if bool(cfg.use_hdsi_s2c):
                            quality = _hdsi_s2c_quality(
                                quality,
                                cls_name=sample.cls_name,
                                s2c_metrics=s2c_metrics,
                                config=cfg,
                            )
                            selection_score = float(quality["hdsi_s2c_selection_score"])
                    elif drr_enabled:
                        quality = _drr_de_lgi_quality(
                            base_quality,
                            energy_target=float(target_energy),
                            energy_actual=actual_energy,
                            background_leakage=background_leakage,
                            spectrum_match=float(last_trace.get("spectrum_match_score", 0.0)),
                            utility_score=float(target_spectrum.get("utility_score", 0.0)),
                            diversity_gain=diversity_gain,
                            residual_recomposition_score=residual_recomposition_score,
                            hdsi_intervention_score=float(target_spectrum.get("hdsi_intervention_score", 0.0)),
                            hdsi_validity_score=float(target_spectrum.get("hdsi_validity_score", 1.0)),
                            config=cfg,
                        )
                        selection_score = float(quality["drr_de_lgi_total"])
                    else:
                        quality = _ucs_de_lgi_quality(
                            base_quality,
                            energy_target=float(target_energy),
                            energy_actual=actual_energy,
                            background_leakage=background_leakage,
                            spectrum_match=float(last_trace.get("spectrum_match_score", 0.0)),
                            utility_score=float(target_spectrum.get("utility_score", 0.0)),
                            diversity_gain=diversity_gain,
                            hdsi_intervention_score=float(target_spectrum.get("hdsi_intervention_score", 0.0)),
                            hdsi_validity_score=float(target_spectrum.get("hdsi_validity_score", 1.0)),
                            config=cfg,
                        )
                        selection_score = float(quality["ucs_de_lgi_total"])
                else:
                    quality = _de_lgi_quality(
                        base_quality,
                        energy_target=float(target_energy),
                        energy_actual=actual_energy,
                        background_leakage=background_leakage,
                    )
                    selection_score = float(quality["de_lgi_total"])
                if bool(cfg.use_hdsi_pd) and isinstance(srt_info.get("hdsi_phase_detail_intervention"), dict):
                    pd_info = srt_info["hdsi_phase_detail_intervention"]
                    pd_enabled = bool(pd_info.get("hdsi_pd_enabled", False)) if isinstance(pd_info, dict) else False
                else:
                    pd_info = {}
                    pd_enabled = False
                if pd_enabled:
                    prototype_records = pd_info.get("prototype_selection", []) if isinstance(pd_info, dict) else []
                    prototype_scores = [
                        float(record.get("prototype_score", 0.0))
                        for record in prototype_records
                        if isinstance(record, dict)
                    ]
                    prototype_score = float(np.mean(prototype_scores)) if prototype_scores else 0.0
                    detail_score = float(
                        np.clip(
                            0.46 * float(last_trace.get("hdsi_pd_detail_gate", 0.0))
                            + 0.34 * float(last_trace.get("hdsi_pd_detail_strength", 0.0)) / max(1e-6, float(cfg.hdsi_pd_late_detail_strength))
                            + 0.20 * float(pd_info.get("hdsi_pd_phase_strength", 0.0)),
                            0.0,
                            1.0,
                        )
                    )
                    pd_selection_score = float(
                        np.clip(
                            0.40 * prototype_score
                            + 0.25 * detail_score
                            + 0.20 * float(np.clip(diversity_gain, 0.0, 1.0))
                            + 0.15 * float(quality.get("outside_guard_score", quality.get("background_score", 0.0))),
                            0.0,
                            1.0,
                        )
                    )
                    pd_weight = 0.0 if str(cfg.method_preset).lower().replace("_", "-") == "spec-srt" else float(np.clip(max(0.12, float(cfg.diversity_weight)), 0.0, 0.35))
                    selection_score = (1.0 - pd_weight) * float(selection_score) + pd_weight * pd_selection_score
                    quality = {
                        **quality,
                        "hdsi_pd_prototype_score": float(prototype_score),
                        "hdsi_pd_detail_score": float(detail_score),
                        "hdsi_pd_selection_score": float(pd_selection_score),
                        "hdsi_pd_selection_weight": float(pd_weight),
                    }
                if str(cfg.method_preset).lower().replace("_", "-") == "e-srt-hdsi-pd":
                    quality = _e_srt_hdsi_pd_quality(
                        quality,
                        cls_name=sample.cls_name,
                        config=cfg,
                    )
                    selection_score = float(quality["e_srt_hdsi_pd_selection_score"])
                candidate_accepted = True
                rejection_reasons: list[str] = []
                if srt_enabled and bool(cfg.srt_strict_s2i_gate) and not bool(s2i_metrics.get("s2i_gate_pass", True)):
                    candidate_accepted = False
                    rejection_reasons.append("s2i_gate_failed")
                if srt_enabled and bool(cfg.use_hdsi_s2c) and bool(cfg.hdsi_s2c_strict_gate) and not bool(quality.get("hdsi_s2c_gate_pass", True)):
                    candidate_accepted = False
                    rejection_reasons.append("hdsi_s2c_gate_failed")
                if srt_enabled and str(cfg.method_preset).lower().replace("_", "-") == "spec-srt" and not bool(quality.get("spec_srt_gate_pass", False)):
                    candidate_accepted = False
                    rejection_reasons.append("spec_srt_gate_failed")
                if str(cfg.method_preset).lower().replace("_", "-") == "e-srt-hdsi-pd" and not bool(quality.get("e_srt_hdsi_pd_gate_pass", True)):
                    candidate_accepted = False
                    rejection_reasons.extend(str(reason) for reason in quality.get("e_srt_hdsi_pd_gate_failures", []))
                if srt_enabled and bool(cfg.use_seca) and bool(cfg.seca_strict) and not bool(seca_info.get("seca_pass", False)):
                    candidate_accepted = False
                    rejection_reasons.append("seca_gate_failed")
                if not candidate_accepted:
                    selection_score = -1.0
                if inherits_source_bbox:
                    planned_generated_bbox_in_expanded = tuple(int(v) for v in local_bbox)
                    planned_generated_bbox_full = tuple(int(v) for v in sample.bbox)
                    generated_bbox_in_expanded = tuple(int(v) for v in local_bbox)
                    generated_bbox_full = tuple(int(v) for v in sample.bbox)
                else:
                    planned_generated_bbox_in_expanded = _scale_bbox_between_sizes(
                        planned_generated_bbox_scaled,
                        working_crop.size,
                        expanded_crop.size,
                    )
                    planned_generated_bbox_full = _offset_bbox(planned_generated_bbox_in_expanded, expanded[:2], original.size)
                    generated_bbox_in_expanded = _scale_bbox_between_sizes(
                        generated_bbox_scaled,
                        working_crop.size,
                        expanded_crop.size,
                    )
                    generated_bbox_full = _offset_bbox(generated_bbox_in_expanded, expanded[:2], original.size)
                row: dict[str, object] = {
                    "target_key": target_key,
                    "source_key": sample.source_stem,
                    "object_index": int(getattr(sample, "object_index", 0)),
                    "object_count": len(getattr(sample, "objects", ()) or ()) or 1,
                    "class": sample.cls_name,
                    "prompt": prompt,
                    "candidate_index": int(candidate_index),
                    "candidate_path": str(candidate_path.resolve()),
                    "selected": False,
                    "method": (
                        "spec_srt_lgi"
                        if str(cfg.method_preset).lower().replace("_", "-") == "spec-srt"
                        else "hdsi_s2c_de_lgi"
                        if srt_enabled and bool(cfg.use_hdsi_s2c)
                        else
                        "e_srt_hdsi_pd_de_lgi"
                        if srt_enabled and bool(cfg.use_e_srt) and bool(cfg.use_hdsi) and bool(cfg.use_hdsi_pd)
                        else "srt_hdsi_pd_de_lgi"
                        if srt_enabled and bool(cfg.use_hdsi) and bool(cfg.use_hdsi_pd)
                        else
                        "e_srt_hdsi_de_lgi"
                        if srt_enabled and bool(cfg.use_e_srt) and bool(cfg.use_hdsi)
                        else "srt_hdsi_de_lgi"
                        if srt_enabled and bool(cfg.use_hdsi)
                        else "e_srt_de_lgi"
                        if srt_enabled and (bool(cfg.use_e_srt) or bool(cfg.use_seca))
                        else "srt_de_lgi"
                        if srt_enabled
                        else "hdsi_de_lgi"
                        if cfg.use_ucs and bool(cfg.use_hdsi)
                        else "drr_de_lgi"
                        if drr_enabled
                        else "ucs_de_lgi"
                        if cfg.use_ucs
                        else "de_lgi"
                    ),
                    "paper_reference": (
                        "Spec-SRT-LGI: spectrum-guided structure residual transport with visibility-constrained projection and source-bbox label inheritance"
                        if str(cfg.method_preset).lower().replace("_", "-") == "spec-srt"
                        else "HDSI-S2C-DE-LGI: hardness-aware defect spectrum intervention with spectrum-to-structure consistency"
                        if srt_enabled and bool(cfg.use_hdsi_s2c)
                        else
                        "E-SRT+HDSI-PD-DE-LGI: evidence-guided structure residual transport with hardness-aware spectrum-phase defect prototype detail intervention"
                        if srt_enabled and bool(cfg.use_e_srt) and bool(cfg.use_hdsi) and bool(cfg.use_hdsi_pd)
                        else "SRT+HDSI-PD-DE-LGI: structure residual transport with hardness-aware spectrum-phase defect prototype detail intervention"
                        if srt_enabled and bool(cfg.use_hdsi) and bool(cfg.use_hdsi_pd)
                        else
                        "E-SRT+HDSI-DE-LGI: evidence-guided structure residual transport with hardness-aware defect spectrum intervention"
                        if srt_enabled and bool(cfg.use_e_srt) and bool(cfg.use_hdsi)
                        else "SRT+HDSI-DE-LGI: structure residual transport with hardness-aware defect spectrum intervention"
                        if srt_enabled and bool(cfg.use_hdsi)
                        else "E-SRT-DE-LGI: evidence-guided structure residual transport with optional SECA labels"
                        if srt_enabled and (bool(cfg.use_e_srt) or bool(cfg.use_seca))
                        else "SRT-DE-LGI: structure-residual transport latent generative intervention"
                        if srt_enabled
                        else "HDSI-DE-LGI: hardness-aware defect spectrum intervention for utility-guided latent inpainting"
                        if cfg.use_ucs and bool(cfg.use_hdsi)
                        else "DRR-DE-LGI: diversity-aware residual recomposition latent inpainting"
                        if drr_enabled
                        else "UCS-DE-LGI: utility-guided causal defect-spectrum latent inpainting"
                        if cfg.use_ucs
                        else "DE-LGI: defect-energy projected latent generative inpainting"
                    ),
                    "spatial_code": {
                        "source_bbox": [int(v) for v in sample.bbox],
                        "expanded_bbox": [int(v) for v in expanded],
                        "defect_bbox_in_expanded": [int(v) for v in local_bbox],
                        "scaled_bbox": [int(v) for v in scaled_local_bbox],
                        "generated_bbox": [int(v) for v in generated_bbox_full],
                        "generated_bbox_in_expanded": [int(v) for v in generated_bbox_in_expanded],
                        "generated_scaled_bbox": [int(v) for v in generated_bbox_scaled],
                        "planned_generated_bbox": [int(v) for v in planned_generated_bbox_full],
                        "planned_generated_bbox_in_expanded": [int(v) for v in planned_generated_bbox_in_expanded],
                        "planned_generated_scaled_bbox": [int(v) for v in planned_generated_bbox_scaled],
                    },
                    "morphology_code": asdict(plan),
                    "structure_residual_transport": srt_info,
                    "energy_projection": {
                        "form": (
                            "s*=argmax H_c,V_c,D; phi'=A(phi_src,s*,bbox); z_{t-1}=z_ctx+Pi_{s*,phi'}(delta_raw+rho_t T(r_src|phi_src->phi') | M_core,M_shell,M_out,visible_s2i)"
                            if str(cfg.method_preset).lower().replace("_", "-") == "spec-srt"
                            else
                            "s*=argmax H_c,V_c,D; phi'=S2C(s*,bbox); z_{t-1}=z_ctx+Pi_{s*,phi'}(delta_raw+rho_t T(r_src|phi_src->phi')); select by S2C(s*,phi',I_gen)"
                            if srt_enabled and bool(cfg.use_hdsi_s2c)
                            else
                            "z_t<-Guide(phi',T_residual) early; z_{t-1}=z_ctx+P_srt(P_hdsi-pd(T_source,T_proto_phase_detail) | phi', target_spectrum_hdsi, M_canvas, M_out); late steps release prototype high-frequency detail only inside phi'"
                            if srt_enabled and bool(cfg.use_hdsi) and bool(cfg.use_hdsi_pd)
                            else
                            "z_t<-Guide(phi',T_residual) early; z_{t-1}=z_ctx+P_srt(delta_raw + rho_t T(z_defect-z_normal | phi_source->phi') | phi', target_spectrum_hdsi, M_canvas, M_out)"
                            if srt_enabled and bool(cfg.use_hdsi)
                            else "z_t<-Guide(phi',T_residual) early; z_{t-1}=z_ctx+P_srt(delta_raw + rho_t T(z_defect-z_normal | phi_source->phi') | phi', target_spectrum, M_canvas, M_out)"
                            if srt_enabled
                            else "z_{t-1}=z_ctx+P_ucs((1-rho_t)delta_raw+rho_t B(R_a^low,R_b^mid,R_c^high) | target_spectrum_far,P_fail,P_syn,M_core,M_shell,M_out)"
                            if drr_enabled and cfg.use_band_recomposition
                            else "z_{t-1}=z_ctx+P_ucs((1-rho_t)delta_raw+rho_t R_donor | target_spectrum,P_fail,P_syn,M_core,M_shell,M_out)"
                            if drr_enabled
                            else "z_{t-1}=z_ctx+P_ucs(delta_raw | target_spectrum,P_fail,P_syn,M_core,M_shell,M_out)"
                            if cfg.use_ucs
                            else "z_{t-1}=z_ctx+P_c(z_raw-z_ctx | M_core,M_shell,M_out)"
                        ),
                        "target_energy": float(target_energy),
                        "profile": asdict(energy_profile),
                        "last_step": last_trace,
                        "eta_mean": float(np.mean(eta_values)) if eta_values else None,
                    },
                    "utility_causal_spectrum": {
                        "target_spectrum": {str(k): _jsonable_manifest_value(v) for k, v in target_spectrum.items()},
                        "source_spectrum": {str(k): _jsonable_manifest_value(v) for k, v in source_spectrum.items()},
                        "distant_spectrum_enabled": bool((drr_enabled or srt_enabled) and cfg.use_distant_spectrum),
                        "hdsi_enabled": bool(cfg.use_hdsi),
                        "hdsi": {
                            "definition": (
                                "select a hard-but-valid source-distant class spectrum for Spec-SRT; prototype detail remains an ablation component"
                                if str(cfg.method_preset).lower().replace("_", "-") == "spec-srt"
                                else "select a hard-but-valid class spectrum and enforce its realization through HDSI-S2C phi'/image consistency"
                                if bool(cfg.use_hdsi_s2c)
                                else "select a hard-but-valid class-spectrum target; HDSI-PD additionally injects real defect prototype phase/detail residuals when enabled without changing the source-bbox label contract"
                            ),
                            "hardness_score": float(target_spectrum.get("hdsi_hardness_score", 0.0)),
                            "validity_score": float(target_spectrum.get("hdsi_validity_score", 1.0)),
                            "source_distance": float(target_spectrum.get("hdsi_source_distance", 0.0)),
                            "history_diversity": float(target_spectrum.get("hdsi_history_diversity", 0.0)),
                            "intervention_score": float(target_spectrum.get("hdsi_intervention_score", 0.0)),
                            "tail_intervention_strength": float(target_spectrum.get("hdsi_tail_intervention_strength", 0.0)),
                            "validity_floor": float(target_spectrum.get("hdsi_validity_floor", cfg.hdsi_min_validity_score)),
                            "source_distance_floor": float(target_spectrum.get("hdsi_source_distance_floor", cfg.hdsi_min_source_distance)),
                            "validity_gate_pass": bool(target_spectrum.get("hdsi_validity_gate_pass", True)),
                            "source_distance_gate_pass": bool(target_spectrum.get("hdsi_source_distance_gate_pass", True)),
                            "hard_valid_gate_pass": bool(target_spectrum.get("hdsi_hard_valid_gate_pass", True)),
                            "hardness_weight": float(cfg.hdsi_hardness_weight),
                            "validity_weight": float(cfg.hdsi_validity_weight),
                            "diversity_weight": float(cfg.hdsi_diversity_weight),
                        },
                        "hdsi_s2c": srt_info.get("spectrum_to_structure_consistency", {}) if bool(cfg.use_hdsi_s2c) else {},
                        "hdsi_pd": srt_info.get("hdsi_phase_detail_intervention", {}) if bool(cfg.use_hdsi_pd) else {},
                        "profile": asdict(spectrum_profile),
                        "diversity_gain": float(diversity_gain),
                        "p_fail_proxy": {
                            "failure_weight": float(target_spectrum.get("failure_weight", 0.0)),
                            "interpretation": "higher values prioritize hard low-contrast, small, elongated, rough, or cluttered defects",
                        },
                        "p_syn_proxy": {
                            "orientation_counts_before": dict(orientation_counts_by_class.get(sample.cls_name, {})),
                        },
                        "drr_residual_recomposition": drr_info,
                    },
                    "latent_inpainting": {
                        "method_preset": str(cfg.method_preset),
                        "resolution": int(cfg.resolution),
                        "steps": int(cfg.num_inference_steps),
                        "eta_min": float(cfg.eta_min),
                        "eta_max": float(cfg.eta_max),
                        "shell_strength": float(cfg.shell_strength),
                        "background_leakage": float(cfg.background_leakage),
                        "background_propagation_sigma": float(cfg.background_propagation_sigma),
                        "boundary_smoothing_sigma": float(cfg.boundary_smoothing_sigma),
                        "use_ucs": bool(cfg.use_ucs),
                        "spectrum_projection_strength": float(cfg.spectrum_projection_strength),
                        "spectrum_orientation_strength": float(cfg.spectrum_orientation_strength),
                        "core_renoise_strength": float(cfg.core_renoise_strength),
                        "use_drr": bool(drr_enabled),
                        "residual_bank_mix": float(cfg.residual_bank_mix),
                        "pseudo_suppression_strength": float(cfg.pseudo_suppression_strength),
                        "structure_jitter": float(cfg.structure_jitter),
                        "diversity_delta_target": float(cfg.diversity_delta_target),
                        "outside_delta_budget": float(cfg.outside_delta_budget),
                        "use_distant_spectrum": bool(cfg.use_distant_spectrum),
                        "distant_spectrum_candidates": int(cfg.distant_spectrum_candidates),
                        "distant_spectrum_weight": float(cfg.distant_spectrum_weight),
                        "use_hdsi": bool(cfg.use_hdsi),
                        "hdsi_candidates": int(cfg.hdsi_candidates),
                        "hdsi_hardness_weight": float(cfg.hdsi_hardness_weight),
                        "hdsi_validity_weight": float(cfg.hdsi_validity_weight),
                        "hdsi_diversity_weight": float(cfg.hdsi_diversity_weight),
                        "hdsi_projection_boost": float(cfg.hdsi_projection_boost),
                        "hdsi_tail_strength": float(cfg.hdsi_tail_strength),
                        "hdsi_min_validity_score": float(cfg.hdsi_min_validity_score),
                        "hdsi_min_source_distance": float(cfg.hdsi_min_source_distance),
                        "use_hdsi_s2c": bool(cfg.use_hdsi_s2c),
                        "hdsi_s2c_structure_weight": float(cfg.hdsi_s2c_structure_weight),
                        "hdsi_s2c_image_weight": float(cfg.hdsi_s2c_image_weight),
                        "hdsi_s2c_min_score": float(cfg.hdsi_s2c_min_score),
                        "hdsi_s2c_strict_gate": bool(cfg.hdsi_s2c_strict_gate),
                        "hdsi_s2c_spectrum_tolerance": float(cfg.hdsi_s2c_spectrum_tolerance),
                        "use_hdsi_pd": bool(cfg.use_hdsi_pd),
                        "hdsi_pd_prototype_count": int(cfg.hdsi_pd_prototype_count),
                        "hdsi_pd_strength": float(cfg.hdsi_pd_strength),
                        "hdsi_pd_phase_strength": float(cfg.hdsi_pd_phase_strength),
                        "hdsi_pd_late_detail_strength": float(cfg.hdsi_pd_late_detail_strength),
                        "hdsi_pd_detail_start": float(cfg.hdsi_pd_detail_start),
                        "hdsi_pd_late_renoise_strength": float(cfg.hdsi_pd_late_renoise_strength),
                        "use_band_recomposition": bool(cfg.use_band_recomposition),
                        "band_donor_count": int(cfg.band_donor_count),
                        "band_recomposition_strength": float(cfg.band_recomposition_strength),
                        "use_srt": bool(srt_enabled),
                        "srt_transport_strength": float(cfg.srt_transport_strength),
                        "srt_bbox_jitter": float(cfg.srt_bbox_jitter),
                        "srt_scale_jitter": float(cfg.srt_scale_jitter),
                        "srt_boundary_roughness": float(cfg.srt_boundary_roughness),
                        "srt_component_jitter": float(cfg.srt_component_jitter),
                        "srt_source_preservation": float(cfg.srt_source_preservation),
                        "srt_bbox_update": bool(cfg.srt_bbox_update),
                        "srt_regeneration_strength": float(cfg.srt_regeneration_strength),
                        "srt_regeneration_until": float(cfg.srt_regeneration_until),
                        "srt_late_texture_mix": float(cfg.srt_late_texture_mix),
                        "srt_s2i_visible_delta_target": float(cfg.srt_s2i_visible_delta_target),
                        "srt_s2i_min_ratio": float(cfg.srt_s2i_min_ratio),
                        "use_e_srt": bool(cfg.use_e_srt),
                        "e_srt_evidence_strength": float(cfg.e_srt_evidence_strength),
                        "e_srt_background_strength": float(cfg.e_srt_background_strength),
                        "e_srt_min_core_energy_ratio": float(cfg.e_srt_min_core_energy_ratio),
                        "e_srt_min_visible_delta": float(cfg.e_srt_min_visible_delta),
                        "e_srt_max_source_similarity": float(cfg.e_srt_max_source_similarity),
                        "e_srt_max_outside_delta": float(cfg.e_srt_max_outside_delta),
                        "e_srt_min_s2i_ratio": float(cfg.e_srt_min_s2i_ratio),
                        "e_srt_min_novelty_score": float(cfg.e_srt_min_novelty_score),
                        "use_seca": bool(cfg.use_seca),
                        "seca_min_coverage": float(cfg.seca_min_coverage),
                        "seca_max_leakage": float(cfg.seca_max_leakage),
                        "seca_min_focus": float(cfg.seca_min_focus),
                        "seca_strict": bool(cfg.seca_strict),
                        "srt_strict_s2i_gate": bool(cfg.srt_strict_s2i_gate),
                        "spec_srt_min_inside_delta": float(cfg.spec_srt_min_inside_delta),
                        "spec_srt_min_change_fraction": float(cfg.spec_srt_min_change_fraction),
                        "spec_srt_max_source_similarity": float(cfg.spec_srt_max_source_similarity),
                        "spec_srt_min_sharpness_ratio": float(cfg.spec_srt_min_sharpness_ratio),
                        "spec_srt_min_source_spectrum_distance": float(cfg.spec_srt_min_source_spectrum_distance),
                    },
                    "mask_stats": {**masks.stats, **({f"canvas_{k}": v for k, v in canvas_masks.stats.items()} if srt_enabled else {})},
                    "quality": quality,
                    "selection_score": selection_score,
                    "candidate_accepted": bool(candidate_accepted),
                    "rejection_reasons": list(rejection_reasons),
                    "paths": {
                        "candidate": str(candidate_path.resolve()),
                        "pseudo_normal_crop": str(pseudo_path.resolve()),
                        "generated_crop": str(gen_crop_path.resolve()),
                        "core_mask": str((mask_dir / f"{target_key}_c{candidate_index:02d}_M_core.png").resolve()),
                        "shell_mask": str((mask_dir / f"{target_key}_c{candidate_index:02d}_M_shell.png").resolve()),
                        "out_mask": str((mask_dir / f"{target_key}_c{candidate_index:02d}_M_out.png").resolve()),
                        "edit_mask": str((mask_dir / f"{target_key}_c{candidate_index:02d}_M_edit.png").resolve()),
                        "canvas_edit_mask": str((mask_dir / f"{target_key}_c{candidate_index:02d}_M_canvas_edit.png").resolve()) if srt_enabled else None,
                        "seca_mask": str(seca_mask_path.resolve()) if seca_mask_path is not None else None,
                        "structure_phi": str((structure_dir / f"{target_key}_c{candidate_index:02d}_phi.png").resolve()) if srt_enabled else None,
                        "source_phi": str((structure_dir / f"{target_key}_c{candidate_index:02d}_phi_source.png").resolve()) if srt_enabled else None,
                    },
                    "label_contract": (
                        "structure-evidence-aligned"
                        if srt_enabled and bool(cfg.use_seca) and bool(seca_info.get("seca_pass", False))
                        else "structure-evidence-failed"
                        if srt_enabled and bool(cfg.use_seca)
                        else "structure-field-derived"
                        if srt_enabled and bool(cfg.srt_bbox_update)
                        else "source-label-inherited"
                    ),
                }
                sample_rows.append(row)
                trace_rows.extend(candidate_traces)
                if candidate_accepted and (best is None or selection_score > best[0]):
                    best = (selection_score, final, annotation_mask, row)

            if best is None:
                rows.extend(sample_rows)
                continue
            _, selected_image, selected_mask, selected_row = best
            selected_row["selected"] = True
            out_path = image_dir / f"{target_key}.png"
            selected_image.save(out_path)
            selected_mask.save(mask_dir / f"{target_key}_selected_M_edit.png")
            selected_row["output_path"] = str(out_path.resolve())
            target_selected = (
                selected_row.get("utility_causal_spectrum", {}).get("target_spectrum")
                if isinstance(selected_row.get("utility_causal_spectrum"), dict)
                else None
            )
            if isinstance(target_selected, dict):
                selected_spectra_by_class.setdefault(sample.cls_name, []).append(dict(target_selected))
                orientation_bin = int(float(target_selected.get("orientation_bin", 0) or 0))
                counts = orientation_counts_by_class.setdefault(sample.cls_name, {})
                counts[orientation_bin] = int(counts.get(orientation_bin, 0)) + 1
            rows.extend(sample_rows)
            outputs.append(out_path)

        score_path = artifacts_dir / "candidate_scores.jsonl"
        score_mode = "a" if score_path.exists() else "w"
        with score_path.open(score_mode, encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        trace_path = artifacts_dir / "energy_trace.jsonl"
        trace_mode = "a" if trace_path.exists() else "w"
        with trace_path.open(trace_mode, encoding="utf-8") as handle:
            for row in trace_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "method": (
                "spec_srt_lgi"
                if str(cfg.method_preset).lower().replace("_", "-") == "spec-srt"
                else "hdsi_s2c_de_lgi"
                if srt_enabled and bool(cfg.use_hdsi_s2c)
                else
                "e_srt_hdsi_pd_de_lgi"
                if srt_enabled and bool(cfg.use_e_srt) and bool(cfg.use_hdsi) and bool(cfg.use_hdsi_pd)
                else "srt_hdsi_pd_de_lgi"
                if srt_enabled and bool(cfg.use_hdsi) and bool(cfg.use_hdsi_pd)
                else
                "e_srt_hdsi_de_lgi"
                if srt_enabled and bool(cfg.use_e_srt) and bool(cfg.use_hdsi)
                else "srt_hdsi_de_lgi"
                if srt_enabled and bool(cfg.use_hdsi)
                else "e_srt_de_lgi"
                if srt_enabled and (bool(cfg.use_e_srt) or bool(cfg.use_seca))
                else "srt_de_lgi"
                if srt_enabled
                else "hdsi_de_lgi"
                if cfg.use_ucs and bool(cfg.use_hdsi)
                else "drr_de_lgi"
                if drr_enabled
                else "ucs_de_lgi"
                if cfg.use_ucs
                else "de_lgi"
            ),
            "model_dir": self.model_dir,
            "generated_images": len(outputs),
            "image_dir": str(image_dir.resolve()),
            "score_path": str(score_path.resolve()),
            "trace_path": str(trace_path.resolve()),
            "prompts_by_class": dict(sorted(prompts_by_class.items())),
            "config": asdict(cfg),
            "p_syn_proxy": {
                cls_name: {str(k): int(v) for k, v in sorted(counts.items())}
                for cls_name, counts in sorted(orientation_counts_by_class.items())
            },
            "innovation": (
                {
                    "hdsi_source_distant_spectrum": "HDSI selects hard-but-valid source-distant class defect spectra before reverse diffusion",
                    "spec_srt_visibility_transport": "Spec-SRT transports source defect residual evidence into a bbox-inherited phi' and validates it with one visibility-constrained projection gate",
                }
                if str(cfg.method_preset).lower().replace("_", "-") == "spec-srt"
                else {
                    "hardness_aware_defect_spectrum_intervention": "HDSI selects hard-but-valid class spectra that are valid under the class prior and distant from the source spectrum",
                    "spectrum_to_structure_consistency": "HDSI-S2C projects the selected spectrum into bbox-constrained phi' fields and re-measures generated pixels against the same target spectrum",
                    "e_srt_feedback": "E-SRT boosts weak phi evidence and dampens off-structure leakage during reverse diffusion",
                    "source_bbox_inheritance": "source bbox labels remain the default label contract while S2C controls intra-box structure",
                    "prototype_detail_ablation": "HDSI-PD is optional and disabled by the HDSI-S2C preset unless requested",
                }
                if bool(cfg.use_hdsi_s2c)
                else {
                    "defect_energy_projection": "latent residuals are projected to class-calibrated defect energy while constraining background leakage",
                    "utility_causal_spectrum_projection": "UCS mode projects latent residuals to a class-wise defect spectrum guided by P_fail and P_syn proxies",
                    "diversity_residual_recomposition": "DRR mode recomposes cross-instance latent defect residuals before causal spectrum projection instead of copying the target defect residual",
                    "source_distant_spectrum": "DRR-V3 can sample target spectra far from the source defect while remaining inside the class prior",
                    "hardness_aware_defect_spectrum_intervention": "HDSI selects hard-but-valid class spectra and lightly intervenes on frequency, roughness, density, elongation and contrast while keeping source-bbox labels by default",
                    "hardness_aware_spectrum_phase_detail_intervention": "HDSI-PD selects hard-but-valid real defect prototypes, transports their residual phase/detail into phi', and releases high-frequency detail only in late denoising steps inside the source-bbox contract",
                    "band_wise_residual_recomposition": "DRR-V3 can compose low, mid and high latent residual bands from different same-class donors",
                    "structure_residual_transport": "SRT mode represents a defect as a transported structure field phi and moves raw latent residuals from the source defect into phi'",
                    "source_bbox_inheritance": "SRT mode inherits source bbox labels by default for stable detector training labels",
                    "structure_derived_labels": "SRT mode can optionally derive generated bbox labels from the transported structure field when srt_bbox_update is enabled",
                    "e_srt_feedback": "E-SRT mode uses latent evidence feedback to boost weak phi evidence and damp off-structure background leakage",
                    "seca_annotation": "SECA derives candidate labels from the agreement between phi' and post-generation visible evidence",
                    "context_preservation": "real context latents remain the anchor at every reverse diffusion step",
                }
            ),
        }
        (output_dir / "de_lgi_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return outputs
