from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn

from ..data.loader import DefectSample


@dataclass
class DRFTResidualField:
    """Pixel-space proxy for the residual-field condition used by DRFT."""

    soft_mask: np.ndarray
    signed_residual: np.ndarray
    orientation_x: np.ndarray
    orientation_y: np.ndarray
    boundary_shell: np.ndarray
    distance: np.ndarray
    family: str
    cls_name: str
    orientation_deg: float
    confidence: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    texture_frequency: Optional[np.ndarray] = None
    visual_residual_evidence: Optional[np.ndarray] = None

    def condition_image(self) -> Image.Image:
        residual = _normalize_signed(self.signed_residual)
        angle = (np.arctan2(self.orientation_y, self.orientation_x) + math.pi) / (2.0 * math.pi)
        mask = np.clip(self.soft_mask, 0.0, 1.0)
        rgb = np.stack([residual, mask, angle], axis=-1)
        return Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")

    def residual_image(self) -> Image.Image:
        return Image.fromarray((_normalize_signed(self.signed_residual) * 255.0).astype(np.uint8), mode="L")

    def mask_image(self) -> Image.Image:
        return Image.fromarray(np.clip(self.soft_mask * 255.0, 0, 255).astype(np.uint8), mode="L")

    def shell_image(self) -> Image.Image:
        return Image.fromarray(np.clip(self.boundary_shell * 255.0, 0, 255).astype(np.uint8), mode="L")

    def confidence_image(self) -> Image.Image:
        value = self.confidence if self.confidence is not None else self.soft_mask
        return Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L")

    def uncertainty_image(self) -> Image.Image:
        value = self.uncertainty if self.uncertainty is not None else np.zeros_like(self.soft_mask)
        return Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L")

    def texture_frequency_image(self) -> Image.Image:
        value = self.texture_frequency if self.texture_frequency is not None else np.zeros_like(self.soft_mask)
        return Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L")

    def visual_residual_evidence_image(self) -> Image.Image:
        value = (
            self.visual_residual_evidence
            if self.visual_residual_evidence is not None
            else self.signed_residual
        )
        return Image.fromarray((_normalize_signed(value) * 255.0).astype(np.uint8), mode="L")


@dataclass(frozen=True)
class DRFTCandidateQuality:
    total: float
    background_score: float
    defect_strength_score: float
    boundary_score: float
    outside_delta: float
    inside_delta: float
    boundary_delta: float


@dataclass(frozen=True)
class DRFTContextContract:
    """Masks that define the context-preserved residual-topology edit region."""

    edit_mask: Image.Image
    core_mask: Image.Image
    shell_mask: Image.Image
    context_mask: Image.Image
    protect_mask: Image.Image
    expanded_bbox: tuple[int, int, int, int]
    stats: dict[str, object]


@dataclass(frozen=True)
class DRFTContextQuality:
    total: float
    context_preservation: float
    background_preservation: float
    protected_preservation: float
    context_delta: float
    outside_edit_delta: float
    protected_delta: float
    edit_coverage: float
    context_coverage: float


@dataclass(frozen=True)
class DRFTCanvasQuality:
    total: float
    defect_evidence_drop: float
    background_preservation: float
    texture_consistency: float
    artifact_score: float
    erase_coverage: float
    candidate_index: int
    sensitivity: float
    core_texture_keep: float
    shell_texture_keep: float


@dataclass
class DRFTResidualPrototype:
    key: str
    cls_name: str
    family: str
    residual_rgb: np.ndarray
    soft_mask: np.ndarray
    orientation_deg: float
    coverage: float


class DRFTResidualPrototypeBank:
    """Class-wise real residual snippets mined from NEU/GC defect samples."""

    def __init__(self, prototypes_by_class: dict[str, list[DRFTResidualPrototype]]) -> None:
        self.prototypes_by_class = {cls: list(items) for cls, items in prototypes_by_class.items()}

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[DefectSample],
        *,
        target_size: tuple[int, int] = (512, 512),
    ) -> "DRFTResidualPrototypeBank":
        rows: dict[str, list[DRFTResidualPrototype]] = {}
        for idx, sample in enumerate(samples):
            try:
                image = Image.open(sample.image_path).convert("RGB")
                orig_size = image.size
                resized = image.resize(target_size, Image.Resampling.LANCZOS)
                scaled_bbox = _scale_bbox(sample.bbox, orig_size, target_size)
                clean, _erase = build_texture_preserving_counterfactual_canvas(resized, scaled_bbox, seed=idx)
                proto = _prototype_from_pair(sample, resized, clean, scaled_bbox, seed=idx)
            except Exception:
                proto = None
            if proto is not None:
                rows.setdefault(sample.cls_name, []).append(proto)
        return cls(rows)

    def sample(
        self,
        cls_name: str,
        *,
        seed: int = 0,
        avoid_key: str = "",
    ) -> Optional[DRFTResidualPrototype]:
        pool = [p for p in self.prototypes_by_class.get(cls_name, []) if p.key != avoid_key]
        if not pool:
            pool = self.prototypes_by_class.get(cls_name, [])
        if not pool:
            pool = [p for items in self.prototypes_by_class.values() for p in items]
        if not pool:
            return None
        return pool[int(seed) % len(pool)]

    def write_manifest(self, path: Path) -> None:
        rows: list[dict[str, object]] = []
        for cls_name, items in sorted(self.prototypes_by_class.items()):
            rows.append(
                {
                    "class": cls_name,
                    "count": len(items),
                    "coverage_mean": float(np.mean([p.coverage for p in items])) if items else 0.0,
                    "prototypes": [
                        {
                            "key": p.key,
                            "family": p.family,
                            "shape": list(p.soft_mask.shape),
                            "orientation_deg": p.orientation_deg,
                            "coverage": p.coverage,
                        }
                        for p in items[:200]
                    ],
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class DRFTConfig:
    in_channels: int = 4
    residual_channels: int = 6
    hidden_size: int = 384
    patch_size: int = 2
    depth: int = 8
    num_heads: int = 6
    num_experts: int = 4
    class_count: int = 16


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=timesteps.device, dtype=timesteps.dtype) * -1
        )
        args = timesteps[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class TimeSpaceResidualExpertBlock(nn.Module):
    """DiT block with soft time/space routing over residual-flow experts."""

    def __init__(self, dim: int, num_heads: int, num_experts: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.router = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, num_experts))

    def forward(self, tokens: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(tokens)
        attn_out, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        tokens = tokens + attn_out

        hidden = self.norm2(tokens)
        time_tokens = time_emb[:, None, :].expand_as(hidden)
        weights = torch.softmax(self.router(torch.cat([hidden, time_tokens], dim=-1)), dim=-1)
        expert_out = torch.stack([expert(hidden) for expert in self.experts], dim=2)
        tokens = tokens + torch.sum(expert_out * weights[..., None], dim=2)
        return tokens


class DefectResidualFlowTransformer(nn.Module):
    """Trainable DRFT backbone for latent residual-flow matching.

    The smoke generator below uses an analytic teacher so we can verify the
    pipeline immediately; this module is the trainable replacement for that
    teacher once clean/defect latent pairs are prepared.
    """

    def __init__(self, cfg: DRFTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch = cfg.patch_size
        self.latent_embed = nn.Conv2d(cfg.in_channels, cfg.hidden_size, kernel_size=patch, stride=patch)
        self.field_embed = nn.Conv2d(cfg.residual_channels, cfg.hidden_size, kernel_size=patch, stride=patch)
        self.time_embed = SinusoidalTimeEmbedding(cfg.hidden_size)
        self.class_embed = nn.Embedding(cfg.class_count, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [TimeSpaceResidualExpertBlock(cfg.hidden_size, cfg.num_heads, cfg.num_experts) for _ in range(cfg.depth)]
        )
        self.out = nn.Linear(cfg.hidden_size, cfg.in_channels * patch * patch)

    def forward(
        self,
        latent: torch.Tensor,
        residual_field: torch.Tensor,
        timestep: torch.Tensor,
        class_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, channels, height, width = latent.shape
        patch = self.cfg.patch_size
        tokens_2d = self.latent_embed(latent) + self.field_embed(residual_field)
        grid_h, grid_w = tokens_2d.shape[-2:]
        tokens = tokens_2d.flatten(2).transpose(1, 2)

        time_emb = self.time_embed(timestep.float())
        if class_id is not None:
            time_emb = time_emb + self.class_embed(class_id)
        tokens = tokens + time_emb[:, None, :]
        for block in self.blocks:
            tokens = block(tokens, time_emb)

        patches = self.out(tokens)
        patches = patches.view(bsz, grid_h, grid_w, channels, patch, patch)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(bsz, channels, height, width)


def flow_matching_loss(
    model: DefectResidualFlowTransformer,
    clean_latents: torch.Tensor,
    defect_latents: torch.Tensor,
    residual_field: torch.Tensor,
    class_id: Optional[torch.Tensor] = None,
    noise_scale: float = 0.08,
) -> torch.Tensor:
    """Rectified-flow objective for clean-to-defect residual transport."""

    bsz = clean_latents.shape[0]
    t = torch.rand(bsz, device=clean_latents.device, dtype=clean_latents.dtype)
    eps = torch.randn_like(clean_latents)
    view = (bsz,) + (1,) * (clean_latents.ndim - 1)
    z_t = (1.0 - t.view(view)) * clean_latents + t.view(view) * defect_latents
    if noise_scale > 0:
        z_t = z_t + noise_scale * torch.sin(math.pi * t).view(view) * eps
    target_velocity = defect_latents - clean_latents
    pred_velocity = model(z_t, residual_field, t, class_id=class_id)
    return torch.nn.functional.mse_loss(pred_velocity.float(), target_velocity.float())


class AnalyticResidualFlowSampler:
    """Deterministic residual-flow teacher used for immediate visual validation."""

    def __init__(self, steps: int = 18) -> None:
        self.steps = max(3, int(steps))

    def sample(
        self,
        clean_canvas: Image.Image,
        field: DRFTResidualField,
        *,
        seed: int = 0,
        return_frames: bool = False,
    ) -> tuple[Image.Image, list[Image.Image]]:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        background = np.asarray(clean_canvas.convert("RGB"), dtype=np.float32)
        shape_residual, texture_residual, boundary_residual = _build_stage_residuals(background, field, rng)
        alpha = np.clip(field.soft_mask + 0.34 * field.boundary_shell, 0.0, 1.0)
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), sigmaX=0.65)
        frames: list[Image.Image] = []

        for step in range(1, self.steps + 1):
            tau = step / float(self.steps)
            residual = (
                _smoothstep(tau, 0.00, 0.42) * shape_residual
                + _smoothstep(tau, 0.30, 0.78) * texture_residual
                + _smoothstep(tau, 0.64, 1.00) * boundary_residual
            )
            current = background + residual[:, :, None] * alpha[:, :, None]
            image = Image.fromarray(np.clip(current, 0, 255).astype(np.uint8))
            if return_frames and (step in {1, max(1, self.steps // 3), max(1, 2 * self.steps // 3), self.steps}):
                frames.append(image)
        return image, frames


def build_defect_residual_field(
    clean_canvas: Image.Image,
    defect_mask: Image.Image,
    cls_name: str,
    *,
    orientation_deg: float = 0.0,
    seed: int = 0,
    prototype: Optional[DRFTResidualPrototype] = None,
) -> DRFTResidualField:
    if defect_mask.size != clean_canvas.size:
        defect_mask = defect_mask.resize(clean_canvas.size, Image.Resampling.BILINEAR)
    mask_u8 = np.asarray(defect_mask.convert("L"), dtype=np.uint8)
    soft = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=1.0).astype(np.float32) / 255.0
    soft = np.clip(soft, 0.0, 1.0)
    family = family_from_class(cls_name)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    core = mask_u8 >= 64
    dilated = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
    shell = np.logical_and(dilated, np.logical_not(core)).astype(np.float32)
    shell = cv2.GaussianBlur(shell, (0, 0), sigmaX=1.1)

    distance = cv2.distanceTransform((core.astype(np.uint8) * 255), cv2.DIST_L2, 3)
    if float(distance.max()) > 0:
        distance = distance / float(distance.max())
    distance = distance.astype(np.float32)

    polarity, amplitude, gamma = _family_residual_params(family, rng)
    modulation = _background_modulation(clean_canvas, rng, texture_weight=0.07, noise_weight=0.05)
    signed_residual = polarity * amplitude * np.power(soft, gamma) * modulation
    if prototype is not None:
        proto_residual = _warp_prototype_residual(prototype, soft, seed=seed)
        signed_residual = 0.42 * signed_residual + 0.58 * proto_residual
    signed_residual = _normalize_field_strength(signed_residual, soft, family)

    angle = math.radians(orientation_deg)
    orientation_x = np.full_like(soft, math.cos(angle), dtype=np.float32)
    orientation_y = np.full_like(soft, math.sin(angle), dtype=np.float32)
    return DRFTResidualField(
        soft_mask=soft.astype(np.float32),
        signed_residual=signed_residual.astype(np.float32),
        orientation_x=orientation_x,
        orientation_y=orientation_y,
        boundary_shell=shell.astype(np.float32),
        distance=distance,
        family=family,
        cls_name=cls_name,
        orientation_deg=float(orientation_deg),
    )


def build_texture_preserving_counterfactual_canvas(
    original: Image.Image,
    scaled_bbox: tuple[int, int, int, int],
    *,
    seed: int = 0,
) -> tuple[Image.Image, Image.Image]:
    """Create a pseudo-normal canvas without erasing the planned defect shape.

    DRFT uses the normal-background branch as a texture carrier. Unlike the
    earlier local editing routines, this routine only suppresses high-confidence existing
    defect evidence in the target box and never falls back to a large geometric
    erase mask, which avoids polygonal inpaint artifacts.
    """

    image = original.convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    height, width = arr.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(scaled_bbox, width, height)
    erase = np.zeros((height, width), dtype=np.uint8)
    if x1 <= x0 or y1 <= y0:
        return image, Image.fromarray(erase, mode="L")

    crop = arr[y0:y1, x0:x1]
    weak = _weak_existing_defect_mask(crop, seed=seed)
    coverage = float(weak.mean()) if weak.size else 0.0
    if 0.004 <= coverage <= 0.32:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
        weak = cv2.dilate(weak, kernel, iterations=1)
        erase[y0:y1, x0:x1] = weak * 255

    if erase.max() == 0:
        return image, Image.fromarray(erase, mode="L")

    inpaint_radius = 2 if max(x1 - x0, y1 - y0) < 180 else 3
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    telea = cv2.inpaint(bgr, erase, inpaint_radius, cv2.INPAINT_TELEA)
    clean = cv2.cvtColor(telea, cv2.COLOR_BGR2RGB).astype(np.float32)

    original_f = arr.astype(np.float32)
    low_original = cv2.GaussianBlur(original_f, (0, 0), sigmaX=4.0)
    low_clean = cv2.GaussianBlur(clean, (0, 0), sigmaX=4.0)
    high_texture = original_f - low_original
    texture_carrier = clean + 0.72 * high_texture

    soft = cv2.GaussianBlur(erase, (0, 0), sigmaX=2.4).astype(np.float32) / 255.0
    soft = np.clip(soft, 0.0, 1.0)
    # Blend mostly with the texture-carried fill, but keep the inpainted low
    # frequency trend so old dark/bright defects are reduced.
    fill = 0.78 * texture_carrier + 0.22 * (clean + (low_original - low_clean))
    blended = original_f * (1.0 - soft[:, :, None]) + fill * soft[:, :, None]
    clean_image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    erase_mask = Image.fromarray(erase, mode="L").filter(ImageFilter.GaussianBlur(radius=1.1)).convert("L")
    return clean_image, erase_mask


def build_adaptive_counterfactual_canvas(
    original: Image.Image,
    scaled_bbox: tuple[int, int, int, int],
    cls_name: str,
    *,
    seed: int = 0,
    candidates: int = 5,
) -> tuple[Image.Image, Image.Image, dict[str, object]]:
    """Create a DRFT-v2 pseudo-normal canvas with class-aware erase strength.

    The v1 canvas deliberately keeps most high-frequency texture. DRFT-v2 uses a
    candidate search where the erase mask is class-aware and the original high
    frequency texture is suppressed inside the erase core while still preserved
    around the transition shell.
    """

    image = original.convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    height, width = arr.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(scaled_bbox, width, height)
    empty = Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L")
    if x1 <= x0 or y1 <= y0:
        return image, empty, {"variant": "drft-v2", "reason": "invalid_bbox"}

    crop = arr[y0:y1, x0:x1]
    family = family_from_class(cls_name)
    settings = _adaptive_canvas_settings(family, candidates, seed)
    best: tuple[float, Image.Image, Image.Image, DRFTCanvasQuality] | None = None

    for idx, setting in enumerate(settings):
        weak = _class_aware_existing_defect_mask(
            crop,
            cls_name,
            seed=seed + idx * 997,
            sensitivity=float(setting["sensitivity"]),
        )
        min_cov, max_cov = _adaptive_coverage_limits(family)
        coverage = float(weak.mean()) if weak.size else 0.0
        if coverage < min_cov:
            # Keep the candidate, but broaden it slightly. Thin cracks and
            # scratches often occupy less than one percent of the box.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            weak = cv2.dilate(weak.astype(np.uint8), kernel, iterations=1)
            coverage = float(weak.mean()) if weak.size else 0.0
        if coverage <= 0.0:
            continue
        if coverage > max_cov:
            weak = _trim_mask_to_coverage(weak, max_cov)
            coverage = float(weak.mean()) if weak.size else 0.0
            if coverage <= 0.0:
                continue

        erase = np.zeros((height, width), dtype=np.uint8)
        kernel_size = int(setting["kernel"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
        weak = cv2.dilate(weak, kernel, iterations=int(setting["dilate"]))
        erase[y0:y1, x0:x1] = np.clip(weak, 0, 1) * 255

        clean_arr = _render_adaptive_canvas_candidate(
            arr,
            erase,
            core_texture_keep=float(setting["core_texture_keep"]),
            shell_texture_keep=float(setting["shell_texture_keep"]),
            inpaint_radius=int(setting["inpaint_radius"]),
        )
        quality = _score_adaptive_counterfactual_canvas(
            arr,
            clean_arr,
            erase,
            (x0, y0, x1, y1),
            family,
            candidate_index=idx,
            sensitivity=float(setting["sensitivity"]),
            core_texture_keep=float(setting["core_texture_keep"]),
            shell_texture_keep=float(setting["shell_texture_keep"]),
        )
        if best is None or quality.total > best[0]:
            clean_image = Image.fromarray(np.clip(clean_arr, 0, 255).astype(np.uint8), mode="RGB")
            erase_image = Image.fromarray(erase, mode="L").filter(ImageFilter.GaussianBlur(radius=1.0)).convert("L")
            best = (quality.total, clean_image, erase_image, quality)

    if best is None:
        clean, erase = build_texture_preserving_counterfactual_canvas(image, scaled_bbox, seed=seed)
        meta = {
            "variant": "drft-v2",
            "fallback": "texture_preserving_v1",
            "class": cls_name,
            "family": family,
        }
        return clean, erase, meta

    _, clean_image, erase_image, quality = best
    meta = {
        "variant": "drft-v2",
        "class": cls_name,
        "family": family,
        "quality": asdict(quality),
    }
    return clean_image, erase_image, meta


def save_drft_artifacts(
    artifact_dir: Path,
    stem: str,
    field: DRFTResidualField,
    frames: Sequence[Image.Image] = (),
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "condition": artifact_dir / f"{stem}_residual_field.png",
        "mask": artifact_dir / f"{stem}_soft_mask.png",
        "residual": artifact_dir / f"{stem}_signed_residual.png",
        "shell": artifact_dir / f"{stem}_boundary_shell.png",
        "meta": artifact_dir / f"{stem}_field.json",
    }
    optional_paths: dict[str, Path] = {}
    if field.confidence is not None:
        optional_paths["confidence"] = artifact_dir / f"{stem}_confidence.png"
    if field.uncertainty is not None:
        optional_paths["uncertainty"] = artifact_dir / f"{stem}_uncertainty.png"
    if field.texture_frequency is not None:
        optional_paths["texture_frequency"] = artifact_dir / f"{stem}_texture_frequency.png"
    if field.visual_residual_evidence is not None:
        optional_paths["visual_residual_evidence"] = artifact_dir / f"{stem}_visual_residual_evidence.png"
    field.condition_image().save(paths["condition"])
    field.mask_image().save(paths["mask"])
    field.residual_image().save(paths["residual"])
    field.shell_image().save(paths["shell"])
    if "confidence" in optional_paths:
        field.confidence_image().save(optional_paths["confidence"])
    if "uncertainty" in optional_paths:
        field.uncertainty_image().save(optional_paths["uncertainty"])
    if "texture_frequency" in optional_paths:
        field.texture_frequency_image().save(optional_paths["texture_frequency"])
    if "visual_residual_evidence" in optional_paths:
        field.visual_residual_evidence_image().save(optional_paths["visual_residual_evidence"])
    paths["meta"].write_text(
        json.dumps(
            {
                "family": field.family,
                "class": field.cls_name,
                "orientation_deg": field.orientation_deg,
                "mask_mean": float(field.soft_mask.mean()),
                "residual_min": float(field.signed_residual.min()),
                "residual_max": float(field.signed_residual.max()),
                "confidence_mean": (
                    float(np.clip(field.confidence, 0.0, 1.0).mean())
                    if field.confidence is not None
                    else None
                ),
                "uncertainty_mean": (
                    float(np.clip(field.uncertainty, 0.0, 1.0).mean())
                    if field.uncertainty is not None
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = {key: str(path.resolve()) for key, path in {**paths, **optional_paths}.items()}
    for idx, frame in enumerate(frames):
        frame_path = artifact_dir / f"{stem}_flow_step{idx:02d}.png"
        frame.save(frame_path)
        result[f"flow_step_{idx:02d}"] = str(frame_path.resolve())
    return result


def append_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_drft_candidate(
    clean_canvas: Image.Image,
    generated: Image.Image,
    field: DRFTResidualField,
) -> DRFTCandidateQuality:
    clean = np.asarray(clean_canvas.convert("RGB"), dtype=np.float32)
    gen = np.asarray(generated.convert("RGB"), dtype=np.float32)
    diff = np.abs(gen - clean).mean(axis=2) / 255.0
    core = field.soft_mask >= 0.24
    shell = field.boundary_shell >= 0.12
    outside = np.logical_and(field.soft_mask < 0.035, field.boundary_shell < 0.03)
    inside_delta = float(diff[core].mean()) if core.any() else 0.0
    outside_delta = float(diff[outside].mean()) if outside.any() else 0.0
    boundary_delta = float(diff[shell].mean()) if shell.any() else 0.0
    background_score = math.exp(-outside_delta / 0.018)
    defect_strength_score = 1.0 - math.exp(-inside_delta / 0.045)
    boundary_score = math.exp(-abs(boundary_delta - inside_delta * 0.38) / 0.055)
    total = 0.44 * background_score + 0.38 * defect_strength_score + 0.18 * boundary_score
    return DRFTCandidateQuality(
        total=float(total),
        background_score=float(background_score),
        defect_strength_score=float(defect_strength_score),
        boundary_score=float(boundary_score),
        outside_delta=float(outside_delta),
        inside_delta=float(inside_delta),
        boundary_delta=float(boundary_delta),
    )


def build_residual_seeded_canvas(
    clean_canvas: Image.Image,
    field: DRFTResidualField,
    *,
    residual_seed_gain: float = 1.0,
) -> tuple[Image.Image, dict[str, object]]:
    """Inject a weak DRFT-v2 signed-residual seed into the inpaint input canvas."""

    gain = max(0.0, float(residual_seed_gain))
    if gain <= 0.0:
        return clean_canvas.copy(), {"enabled": False, "residual_seed_gain": 0.0}
    width, height = clean_canvas.size
    arr = np.asarray(clean_canvas.convert("RGB"), dtype=np.float32)
    soft = _resize_float_field(field.soft_mask, (width, height))
    signed = _resize_numeric_field(field.signed_residual, (width, height))
    core = soft > 0.08
    if not core.any():
        return clean_canvas.copy(), {"enabled": False, "residual_seed_gain": gain, "reason": "empty_core"}

    scale = max(1.0, float(np.percentile(np.abs(signed[core]), 88)))
    normalized = np.clip(signed / scale, -1.85, 1.85)
    amplitude = _family_seed_amplitude(field.family)
    alpha = np.power(np.clip(soft, 0.0, 1.0), 0.78)
    confidence_guided = False
    if field.confidence is not None:
        confidence = _resize_float_field(field.confidence, (width, height))
        alpha *= np.clip(0.18 + 0.82 * confidence, 0.0, 1.0)
        confidence_guided = True
    else:
        confidence = np.clip(soft, 0.0, 1.0)
    if field.uncertainty is not None:
        uncertainty = _resize_float_field(field.uncertainty, (width, height))
        alpha *= np.clip(1.0 - 0.62 * uncertainty, 0.20, 1.0)
    else:
        uncertainty = np.zeros_like(soft, dtype=np.float32)
    delta = normalized * amplitude * gain * alpha
    seeded = np.clip(arr + delta[:, :, None], 0, 255)
    meta = {
        "enabled": True,
        "residual_seed_gain": float(gain),
        "confidence_guided": bool(confidence_guided),
        "family": field.family,
        "amplitude": float(amplitude),
        "normalization_scale": float(scale),
        "confidence_mean": float(np.mean(confidence[core])) if core.any() else 0.0,
        "uncertainty_mean": float(np.mean(uncertainty[core])) if core.any() else 0.0,
        "inside_seed_delta_mean": float(np.mean(np.abs(delta[core]))),
        "inside_seed_delta_p95": float(np.percentile(np.abs(delta[core]), 95)),
    }
    return Image.fromarray(seeded.astype(np.uint8), mode="RGB"), meta


def build_context_preserved_drft_contract(
    defect_mask: Image.Image,
    field: DRFTResidualField,
    target_bbox: tuple[int, int, int, int],
    target_size: tuple[int, int],
    *,
    protect_mask: Optional[Image.Image] = None,
    context_dilation: float = 1.65,
    shell_weight: float = 0.55,
    core_edit_weight: float = 1.0,
    core_threshold: float = 0.10,
    edit_threshold: float = 0.03,
    variant_name: str = "drft-v2-context",
) -> DRFTContextContract:
    """Build the DRFT-v2 context-preservation mask contract.

    The returned edit mask is fed to the inpainting pipeline. Stable Diffusion
    inpainting keeps latent values outside that mask during denoising, so this
    mask is the step-wise context injection point rather than a post-hoc blend.
    """

    width, height = target_size
    field_soft = _resize_float_field(field.soft_mask, target_size)
    field_shell = _resize_float_field(field.boundary_shell, target_size)
    mask_soft = _mask_to_float(defect_mask, target_size)
    core = np.clip(np.maximum(mask_soft, field_soft), 0.0, 1.0)
    shell = np.clip(field_shell, 0.0, 1.0)
    protected = (
        _mask_to_float(protect_mask, target_size)
        if protect_mask is not None
        else np.zeros((height, width), dtype=np.float32)
    )
    protected_binary = protected >= 0.5

    base_bbox = _merge_bboxes(
        _clip_bbox(target_bbox, width, height),
        _bbox_from_mask(core > edit_threshold, width, height),
    )
    expanded_bbox = _expand_bbox(base_bbox, width, height, max(1.0, float(context_dilation)))
    x0, y0, x1, y1 = expanded_bbox
    window = np.zeros((height, width), dtype=np.float32)
    window[y0:y1, x0:x1] = 1.0

    core_binary = core >= max(float(core_threshold), edit_threshold)
    core_mask = np.maximum(core * 0.72, core_binary.astype(np.float32) * max(0.0, float(core_edit_weight))) * window
    shell_mask = shell * window
    shell_edit = shell_mask * max(0.0, float(shell_weight))
    if float(np.max(shell_edit)) > 0.0:
        shell_edit = cv2.GaussianBlur(shell_edit.astype(np.float32), (0, 0), sigmaX=0.75)
    edit = np.maximum(core_mask, shell_edit)
    edit = np.clip(edit * window, 0.0, 1.0)
    edit[protected_binary] = 0.0
    core_mask[protected_binary] = 0.0
    shell_mask[protected_binary] = 0.0

    edit_binary = edit > edit_threshold
    context = window.copy()
    context[edit_binary] = 0.0
    context[protected_binary] = 0.0

    denom = float(max(1, width * height))
    edit_pixels = float(np.count_nonzero(edit_binary))
    context_pixels = float(np.count_nonzero(context > 0.5))
    protected_pixels = float(np.count_nonzero(protected_binary))
    stats: dict[str, object] = {
        "variant": str(variant_name),
        "target_bbox": [int(v) for v in _clip_bbox(target_bbox, width, height)],
        "expanded_bbox": [int(v) for v in expanded_bbox],
        "context_dilation": float(context_dilation),
        "shell_weight": float(shell_weight),
        "core_edit_weight": float(core_edit_weight),
        "core_threshold": float(core_threshold),
        "edit_threshold": float(edit_threshold),
        "core_coverage": float(np.count_nonzero(core_mask > edit_threshold) / denom),
        "hard_core_coverage": float(np.count_nonzero(core_binary * (window > 0.5)) / denom),
        "shell_coverage": float(np.count_nonzero(shell_mask > edit_threshold) / denom),
        "edit_coverage": float(edit_pixels / denom),
        "context_coverage": float(context_pixels / denom),
        "protected_coverage": float(protected_pixels / denom),
        "context_to_edit_ratio": float(context_pixels / max(1.0, edit_pixels)),
        "edit_within_expanded_bbox": True,
        "label_contract": "source-label-inherited",
    }
    return DRFTContextContract(
        edit_mask=_float_mask_image(edit),
        core_mask=_float_mask_image(core_mask),
        shell_mask=_float_mask_image(shell_mask),
        context_mask=_float_mask_image(context),
        protect_mask=_float_mask_image(protected),
        expanded_bbox=expanded_bbox,
        stats=stats,
    )


def score_drft_context_contract(
    original: Image.Image,
    generated: Image.Image,
    contract: DRFTContextContract,
) -> DRFTContextQuality:
    """Measure whether non-edit context stayed stable under the contract."""

    size = original.size
    orig = np.asarray(original.convert("RGB"), dtype=np.float32)
    gen = np.asarray(generated.resize(size, Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32)
    diff = np.abs(gen - orig).mean(axis=2) / 255.0
    edit = _mask_to_float(contract.edit_mask, size) > 0.03
    context = _mask_to_float(contract.context_mask, size) > 0.5
    protected = _mask_to_float(contract.protect_mask, size) > 0.5
    outside_edit = np.logical_not(edit)

    context_delta = _masked_mean(diff, context)
    outside_delta = _masked_mean(diff, outside_edit)
    protected_delta = _masked_mean(diff, protected)
    context_preservation = math.exp(-context_delta / 0.015)
    background_preservation = math.exp(-outside_delta / 0.018)
    protected_preservation = math.exp(-protected_delta / 0.010) if protected.any() else 1.0
    total = 0.44 * context_preservation + 0.36 * background_preservation + 0.20 * protected_preservation
    return DRFTContextQuality(
        total=float(total),
        context_preservation=float(context_preservation),
        background_preservation=float(background_preservation),
        protected_preservation=float(protected_preservation),
        context_delta=float(context_delta),
        outside_edit_delta=float(outside_delta),
        protected_delta=float(protected_delta),
        edit_coverage=float(np.count_nonzero(edit) / float(max(1, edit.size))),
        context_coverage=float(np.count_nonzero(context) / float(max(1, context.size))),
    )


def family_from_class(cls_name: str) -> str:
    cls = cls_name.lower()
    if "scratch" in cls:
        return "scratch"
    if "crazing" in cls or "crack" in cls:
        return "crack"
    if "pit" in cls or "hole" in cls or "blow" in cls:
        return "pitted"
    if "inclusion" in cls:
        return "inclusion"
    if "patch" in cls:
        return "patch"
    if "scale" in cls or "rolled" in cls:
        return "scale"
    return "blob"


def build_class_aware_defect_residual_field(
    clean_canvas: Image.Image,
    defect_mask: Image.Image,
    cls_name: str,
    *,
    orientation_deg: float = 0.0,
    seed: int = 0,
    prototype: Optional[DRFTResidualPrototype] = None,
) -> DRFTResidualField:
    """Build the DRFT-v2 residual condition with class-specific intensity priors."""

    field = build_defect_residual_field(
        clean_canvas,
        defect_mask,
        cls_name,
        orientation_deg=orientation_deg,
        seed=seed,
        prototype=prototype,
    )
    rng = np.random.default_rng(int(seed + 4099) & 0xFFFFFFFF)
    background = np.asarray(clean_canvas.convert("RGB"), dtype=np.float32)
    family = field.family
    soft = np.clip(field.soft_mask.astype(np.float32), 0.0, 1.0)
    shell = np.clip(field.boundary_shell.astype(np.float32), 0.0, 1.0)

    if family in {"scratch", "crack"}:
        sharpened = np.clip(0.62 * np.power(soft, 0.88) + 0.38 * np.power(soft, 1.85), 0.0, 1.0)
        field.soft_mask = sharpened.astype(np.float32)
        soft = sharpened
    elif family == "pitted":
        field.soft_mask = np.clip(0.72 * soft + 0.28 * cv2.GaussianBlur(soft, (0, 0), sigmaX=0.45), 0.0, 1.0)
        soft = field.soft_mask
    elif family in {"patch", "scale"}:
        field.soft_mask = np.clip(cv2.GaussianBlur(soft, (0, 0), sigmaX=0.75), 0.0, 1.0)
        soft = field.soft_mask

    class_residual = _class_specific_signed_residual(background, field, rng)
    blend = 0.62 if prototype is not None else 0.48
    signed = (1.0 - blend) * field.signed_residual + blend * class_residual
    signed = _normalize_field_strength(signed.astype(np.float32), soft, family)
    field.signed_residual = signed.astype(np.float32)
    return field


def _class_specific_signed_residual(
    background: np.ndarray,
    field: DRFTResidualField,
    rng: np.random.Generator,
) -> np.ndarray:
    soft = np.clip(field.soft_mask.astype(np.float32), 0.0, 1.0)
    shell = np.clip(field.boundary_shell.astype(np.float32), 0.0, 1.0)
    family = field.family
    if family == "scratch":
        line = np.power(soft, 1.75)
        grain = _oriented_grain(background, field, rng, strength=5.5, freq=0.080)
        return 24.0 * line + 0.65 * grain * line - 4.5 * shell
    if family == "crack":
        line = np.power(soft, 1.52)
        grain = _oriented_grain(background, field, rng, strength=-4.8, freq=0.060)
        return -20.0 * line + 0.70 * grain * line + 3.2 * shell
    if family == "pitted":
        body = np.power(soft, 1.22)
        highlight = np.clip(np.roll(np.roll(body, -1, axis=0), 1, axis=1) - 0.40 * body, 0.0, 1.0)
        grain = rng.normal(0.0, 1.0, size=soft.shape).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.55)
        return -17.0 * body * np.clip(1.0 + 0.16 * grain, 0.68, 1.26) + 3.6 * highlight
    if family == "inclusion":
        body = np.power(soft, 1.34)
        stain = rng.normal(0.0, 1.0, size=soft.shape).astype(np.float32)
        stain = cv2.GaussianBlur(stain, (0, 0), sigmaX=2.5)
        return -16.5 * body * np.clip(0.95 + 0.13 * stain, 0.72, 1.16) + 2.2 * shell
    if family == "patch":
        low = rng.normal(0.0, 1.0, size=soft.shape).astype(np.float32)
        low = cv2.GaussianBlur(low, (0, 0), sigmaX=5.0)
        low = low / max(1.0, float(low.std()))
        return -13.5 * np.power(soft, 1.04) * np.clip(0.96 + 0.10 * low, 0.78, 1.18) - 1.8 * shell
    if family == "scale":
        grain = _oriented_grain(background, field, rng, strength=4.0, freq=0.035)
        polarity = -1.0 if rng.random() < 0.72 else 1.0
        return polarity * 13.0 * np.power(soft, 1.08) + 0.60 * grain * soft - 2.8 * shell
    return field.signed_residual.astype(np.float32)


def _build_stage_residuals(
    background: np.ndarray,
    field: DRFTResidualField,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signed = field.signed_residual
    soft = field.soft_mask
    shell = field.boundary_shell
    shape = cv2.GaussianBlur(signed, (0, 0), sigmaX=3.4)

    if field.family == "scratch":
        texture = _oriented_grain(background, field, rng, strength=7.5, freq=0.075) * np.power(soft, 1.25)
        texture += 18.0 * np.power(soft, 1.85)
        boundary = -4.2 * shell + 2.0 * cv2.GaussianBlur(soft, (0, 0), sigmaX=0.55)
    elif field.family == "crack":
        texture = _oriented_grain(background, field, rng, strength=-6.0, freq=0.055) * np.power(soft, 1.35)
        texture += -11.0 * np.power(soft, 1.55)
        boundary = 3.0 * shell
    elif field.family == "pitted":
        pits = np.power(soft, 1.30)
        highlight = np.clip(np.roll(np.roll(pits, -1, axis=0), 1, axis=1) - pits * 0.38, 0.0, 1.0)
        grain = rng.normal(0.0, 1.0, size=soft.shape).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.55)
        texture = (-10.0 * pits * np.clip(1.0 + 0.18 * grain, 0.62, 1.28)) + 3.2 * highlight
        boundary = 1.6 * shell
    elif field.family == "inclusion":
        body = np.power(soft, 1.42)
        stain = rng.normal(0.0, 1.0, size=soft.shape).astype(np.float32)
        stain = cv2.GaussianBlur(stain, (0, 0), sigmaX=2.2)
        texture = -10.5 * body * np.clip(0.92 + 0.14 * stain, 0.70, 1.15)
        boundary = 1.8 * shell
    elif field.family == "patch":
        body = np.power(soft, 1.08)
        low = rng.normal(0.0, 1.0, size=soft.shape).astype(np.float32)
        low = cv2.GaussianBlur(low, (0, 0), sigmaX=4.5)
        low = low / max(1.0, float(low.std()))
        texture = -10.0 * body * np.clip(0.94 + 0.10 * low, 0.78, 1.18)
        texture += _oriented_grain(background, field, rng, strength=2.2, freq=0.020) * body
        boundary = -1.6 * shell
    elif field.family == "scale":
        texture = -8.0 * np.power(soft, 1.12) + _oriented_grain(background, field, rng, strength=3.0, freq=0.035) * soft
        boundary = -3.0 * shell
    else:
        texture = signed * 0.45 + _background_modulation_array(background, rng, 0.08, 0.08) * 4.0 * soft
        boundary = -2.5 * shell

    return shape.astype(np.float32), texture.astype(np.float32), boundary.astype(np.float32)


def _oriented_grain(
    background: np.ndarray,
    field: DRFTResidualField,
    rng: np.random.Generator,
    *,
    strength: float,
    freq: float,
) -> np.ndarray:
    height, width = field.soft_mask.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    direction = field.orientation_x * xx + field.orientation_y * yy
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    wave = np.sin(direction * freq + phase)
    noise = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=0.75)
    texture = _texture_from_background(background)
    return strength * (0.55 * wave + 0.22 * noise + 0.23 * texture)


def _background_modulation(clean_canvas: Image.Image, rng: np.random.Generator, *, texture_weight: float, noise_weight: float) -> np.ndarray:
    arr = np.asarray(clean_canvas.convert("RGB"), dtype=np.float32)
    return _background_modulation_array(arr, rng, texture_weight, noise_weight)


def _background_modulation_array(arr: np.ndarray, rng: np.random.Generator, texture_weight: float, noise_weight: float) -> np.ndarray:
    texture = _texture_from_background(arr)
    noise = rng.normal(0.0, 1.0, size=texture.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.0)
    return np.clip(1.0 + texture_weight * texture + noise_weight * noise, 0.70, 1.28)


def _texture_from_background(arr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0)
    texture = gray - low
    return texture / max(1.0, float(texture.std()))


def _family_residual_params(family: str, rng: np.random.Generator) -> tuple[float, float, float]:
    if family == "scratch":
        return 1.0, float(rng.uniform(18.0, 27.0)), 1.52
    if family == "crack":
        return -1.0, float(rng.uniform(13.0, 22.0)), 1.35
    if family == "pitted":
        return -1.0, float(rng.uniform(9.0, 16.0)), 1.18
    if family == "inclusion":
        return -1.0, float(rng.uniform(9.0, 15.0)), 1.30
    if family == "patch":
        return -1.0 if rng.random() < 0.82 else 1.0, float(rng.uniform(7.0, 14.0)), 1.05
    if family == "scale":
        return -1.0 if rng.random() < 0.72 else 1.0, float(rng.uniform(8.0, 16.0)), 1.05
    return -1.0 if rng.random() < 0.65 else 1.0, float(rng.uniform(8.0, 18.0)), 1.12


def _normalize_field_strength(signed: np.ndarray, soft: np.ndarray, family: str) -> np.ndarray:
    core = soft > 0.20
    if not core.any():
        return signed.astype(np.float32)
    mean_abs = float(np.mean(np.abs(signed[core])))
    if family in {"scratch", "crack"}:
        target = 23.0
        cap = 38.0
    elif family == "pitted":
        target = 16.0
        cap = 28.0
    elif family in {"inclusion", "patch", "scale"}:
        target = 18.0
        cap = 31.0
    else:
        target = 17.0
        cap = 30.0
    if mean_abs < 1.0:
        return signed.astype(np.float32)
    gain = min(2.4, max(0.72, target / mean_abs))
    signed = signed * gain
    return np.clip(signed, -cap, cap).astype(np.float32)


def _normalize_structure_field_strength(signed: np.ndarray, soft: np.ndarray, family: str) -> np.ndarray:
    core = soft > 0.08
    if not core.any():
        return signed.astype(np.float32)
    signed = _zero_mean_inside(signed.astype(np.float32), soft)
    mean_abs = float(np.mean(np.abs(signed[core])))
    if mean_abs < 1e-5:
        return signed.astype(np.float32)
    if family in {"scratch", "crack"}:
        target, cap = 18.0, 30.0
    elif family == "pitted":
        target, cap = 17.0, 29.0
    elif family in {"inclusion", "patch", "scale"}:
        target, cap = 16.0, 28.0
    else:
        target, cap = 15.0, 26.0
    gain = min(2.15, max(0.58, target / mean_abs))
    signed = _zero_mean_inside(signed * gain, soft)
    return np.clip(signed, -cap, cap).astype(np.float32)


def _zero_mean_inside(arr: np.ndarray, soft: np.ndarray) -> np.ndarray:
    result = np.asarray(arr, dtype=np.float32).copy()
    mask = soft > 0.04
    if not mask.any():
        return result
    weights = np.clip(soft[mask].astype(np.float32), 0.0, 1.0)
    denom = max(1e-6, float(weights.sum()))
    mean = float((result[mask] * weights).sum() / denom)
    result[mask] -= mean
    result *= np.clip(soft, 0.0, 1.0)
    return result.astype(np.float32)


def _weighted_mean(arr: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> float:
    if not mask.any():
        return 0.0
    values = np.asarray(arr, dtype=np.float32)[mask]
    if weights is None:
        return float(values.mean())
    w = np.asarray(weights, dtype=np.float32)[mask]
    denom = max(1e-6, float(w.sum()))
    return float((values * w).sum() / denom)


def _gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    src = np.asarray(arr, dtype=np.float32)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    scale = max(1e-6, float(np.percentile(mag, 98))) if mag.size else 1.0
    return np.clip(mag / scale, 0.0, 1.0).astype(np.float32)


def _edge_map(arr: np.ndarray) -> np.ndarray:
    src = np.asarray(arr, dtype=np.float32)
    if src.size == 0:
        return np.zeros(src.shape, dtype=bool)
    scale = max(1.0, float(np.percentile(src, 98)))
    u8 = np.clip(src / scale * 255.0, 0, 255).astype(np.uint8)
    return cv2.Canny(u8, 40, 110) > 0


def _component_change_score(changed: np.ndarray, core: np.ndarray) -> tuple[float, float]:
    if not changed.any() or not core.any():
        return 0.0, 0.0
    labels, stats = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)[1:3]
    if stats.shape[0] <= 1:
        return 0.0, 0.0
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    largest = float(component_areas.max())
    core_pixels = float(max(1, np.count_nonzero(core)))
    largest_fraction = largest / core_pixels
    total_changed = float(np.count_nonzero(changed)) / core_pixels
    if largest_fraction < 0.015:
        score = largest_fraction / 0.015
    elif largest_fraction > 0.82:
        score = math.exp(-(largest_fraction - 0.82) / 0.18)
    else:
        score = 1.0
    fragmentation = min(1.0, largest / max(1.0, float(component_areas.sum())))
    score = float(score * (0.72 + 0.28 * fragmentation) * (1.0 - math.exp(-total_changed / 0.10)))
    return score, float(largest_fraction)


def _masked_sharpness_ratio(gen_gray: np.ndarray, ref_gray: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 1.0
    gen_lap = cv2.Laplacian(gen_gray.astype(np.float32), cv2.CV_32F)
    ref_lap = cv2.Laplacian(ref_gray.astype(np.float32), cv2.CV_32F)
    gen_var = float(np.var(gen_lap[mask]))
    ref_var = float(np.var(ref_lap[mask]))
    return float(gen_var / max(1.0, ref_var))


def _prototype_from_pair(
    sample: DefectSample,
    image: Image.Image,
    clean: Image.Image,
    scaled_bbox: tuple[int, int, int, int],
    *,
    seed: int = 0,
) -> Optional[DRFTResidualPrototype]:
    img = np.asarray(image.convert("RGB"), dtype=np.float32)
    base = np.asarray(clean.convert("RGB"), dtype=np.float32)
    height, width = img.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(scaled_bbox, width, height)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1]
    diff = img - base
    diff_crop = diff[y0:y1, x0:x1]
    weak = _weak_existing_defect_mask(np.clip(crop, 0, 255).astype(np.uint8), seed=seed).astype(np.float32)
    residual_mag = np.abs(diff_crop).mean(axis=2)
    if residual_mag.size and float(residual_mag.max()) > 1.0:
        mag_mask = residual_mag > max(3.0, float(np.percentile(residual_mag, 76)))
        weak = np.maximum(weak, mag_mask.astype(np.float32))
    weak = cv2.GaussianBlur(weak, (0, 0), sigmaX=1.2)
    weak = np.clip(weak, 0.0, 1.0)
    coverage = float((weak > 0.18).mean()) if weak.size else 0.0
    if coverage < 0.003 or coverage > 0.55:
        return None
    gray_residual = diff_crop.mean(axis=2)
    orientation = _estimate_orientation_deg(weak)
    return DRFTResidualPrototype(
        key=getattr(sample, "target_key", sample.image_path.stem),
        cls_name=sample.cls_name,
        family=family_from_class(sample.cls_name),
        residual_rgb=diff_crop.astype(np.float32),
        soft_mask=weak.astype(np.float32),
        orientation_deg=float(orientation),
        coverage=coverage,
    )


def _warp_prototype_residual(
    prototype: DRFTResidualPrototype,
    target_soft_mask: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    target = np.zeros_like(target_soft_mask, dtype=np.float32)
    coords = np.argwhere(target_soft_mask > 0.05)
    if coords.size == 0:
        return target
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    box_h = max(1, int(y1 - y0))
    box_w = max(1, int(x1 - x0))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    proto_residual = prototype.residual_rgb.mean(axis=2)
    proto_mask = prototype.soft_mask
    if rng.random() < 0.5:
        proto_residual = np.flip(proto_residual, axis=1)
        proto_mask = np.flip(proto_mask, axis=1)
    if rng.random() < 0.25:
        proto_residual = np.flip(proto_residual, axis=0)
        proto_mask = np.flip(proto_mask, axis=0)
    proto_resized = cv2.resize(proto_residual, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.resize(proto_mask, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = np.clip(mask_resized, 0.0, 1.0)

    target_crop = target_soft_mask[y0:y1, x0:x1]
    proto_scale = float(rng.uniform(0.82, 1.18))
    proto_resized = proto_resized * proto_scale
    proto_resized = cv2.GaussianBlur(proto_resized, (0, 0), sigmaX=0.45)
    mixed_mask = np.clip(0.55 * target_crop + 0.45 * mask_resized, 0.0, 1.0)
    target[y0:y1, x0:x1] = proto_resized * mixed_mask
    return target


def _estimate_orientation_deg(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0.18)
    if xs.size < 4:
        return 0.0
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = coords.T @ coords / max(1, coords.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    vec = vecs[:, int(np.argmax(vals))]
    return float(math.degrees(math.atan2(float(vec[1]), float(vec[0]))))


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_w, src_h = source_size
    dst_w, dst_h = target_size
    x0, y0, x1, y1 = bbox
    return _clip_bbox(
        (
            int(x0 * dst_w / max(src_w, 1)),
            int(y0 * dst_h / max(src_h, 1)),
            int(x1 * dst_w / max(src_w, 1)),
            int(y1 * dst_h / max(src_h, 1)),
        ),
        dst_w,
        dst_h,
    )


def _adaptive_canvas_settings(
    family: str,
    candidates: int,
    seed: int,
) -> list[dict[str, float | int]]:
    candidates = max(1, int(candidates))
    if family in {"scratch", "crack"}:
        base = [
            {"sensitivity": 0.92, "core_texture_keep": 0.04, "shell_texture_keep": 0.30, "kernel": 3, "dilate": 1, "inpaint_radius": 2},
            {"sensitivity": 1.08, "core_texture_keep": 0.08, "shell_texture_keep": 0.40, "kernel": 3, "dilate": 2, "inpaint_radius": 2},
            {"sensitivity": 1.22, "core_texture_keep": 0.14, "shell_texture_keep": 0.48, "kernel": 5, "dilate": 1, "inpaint_radius": 3},
            {"sensitivity": 0.78, "core_texture_keep": 0.02, "shell_texture_keep": 0.24, "kernel": 3, "dilate": 1, "inpaint_radius": 2},
            {"sensitivity": 1.36, "core_texture_keep": 0.18, "shell_texture_keep": 0.55, "kernel": 5, "dilate": 2, "inpaint_radius": 3},
        ]
    elif family in {"pitted", "inclusion"}:
        base = [
            {"sensitivity": 0.92, "core_texture_keep": 0.06, "shell_texture_keep": 0.34, "kernel": 3, "dilate": 1, "inpaint_radius": 2},
            {"sensitivity": 1.10, "core_texture_keep": 0.10, "shell_texture_keep": 0.42, "kernel": 5, "dilate": 1, "inpaint_radius": 2},
            {"sensitivity": 1.28, "core_texture_keep": 0.16, "shell_texture_keep": 0.52, "kernel": 5, "dilate": 2, "inpaint_radius": 3},
            {"sensitivity": 0.78, "core_texture_keep": 0.03, "shell_texture_keep": 0.28, "kernel": 3, "dilate": 1, "inpaint_radius": 2},
            {"sensitivity": 1.44, "core_texture_keep": 0.20, "shell_texture_keep": 0.58, "kernel": 7, "dilate": 1, "inpaint_radius": 3},
        ]
    else:
        base = [
            {"sensitivity": 0.86, "core_texture_keep": 0.08, "shell_texture_keep": 0.36, "kernel": 5, "dilate": 1, "inpaint_radius": 3},
            {"sensitivity": 1.04, "core_texture_keep": 0.14, "shell_texture_keep": 0.46, "kernel": 5, "dilate": 2, "inpaint_radius": 3},
            {"sensitivity": 1.22, "core_texture_keep": 0.20, "shell_texture_keep": 0.56, "kernel": 7, "dilate": 1, "inpaint_radius": 3},
            {"sensitivity": 0.72, "core_texture_keep": 0.04, "shell_texture_keep": 0.30, "kernel": 3, "dilate": 1, "inpaint_radius": 2},
            {"sensitivity": 1.42, "core_texture_keep": 0.24, "shell_texture_keep": 0.62, "kernel": 7, "dilate": 2, "inpaint_radius": 3},
        ]
    if candidates <= len(base):
        return base[:candidates]
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    extra: list[dict[str, float | int]] = []
    while len(base) + len(extra) < candidates:
        extra.append(
            {
                "sensitivity": float(rng.uniform(0.78, 1.42)),
                "core_texture_keep": float(rng.uniform(0.03, 0.22)),
                "shell_texture_keep": float(rng.uniform(0.28, 0.62)),
                "kernel": int(rng.choice([3, 5, 7])),
                "dilate": int(rng.choice([1, 2])),
                "inpaint_radius": int(rng.choice([2, 3])),
            }
        )
    return base + extra


def _adaptive_coverage_limits(family: str) -> tuple[float, float]:
    if family in {"scratch", "crack"}:
        return 0.0015, 0.42
    if family in {"pitted", "inclusion"}:
        return 0.0025, 0.46
    if family in {"patch", "scale"}:
        return 0.0060, 0.55
    return 0.0030, 0.50


def _render_adaptive_canvas_candidate(
    arr: np.ndarray,
    erase: np.ndarray,
    *,
    core_texture_keep: float,
    shell_texture_keep: float,
    inpaint_radius: int,
) -> np.ndarray:
    if erase.max() == 0:
        return arr.astype(np.float32)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    telea = cv2.inpaint(bgr, erase, max(1, int(inpaint_radius)), cv2.INPAINT_TELEA)
    clean = cv2.cvtColor(telea, cv2.COLOR_BGR2RGB).astype(np.float32)

    original_f = arr.astype(np.float32)
    low_original = cv2.GaussianBlur(original_f, (0, 0), sigmaX=4.0)
    low_clean = cv2.GaussianBlur(clean, (0, 0), sigmaX=4.0)
    high_texture = original_f - low_original

    core = (erase > 0).astype(np.uint8)
    shell = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    shell = np.clip(shell - core, 0, 1).astype(np.float32)
    core_soft = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=1.2)
    shell_soft = cv2.GaussianBlur(shell, (0, 0), sigmaX=2.0)
    keep = np.clip(core_texture_keep * core_soft + shell_texture_keep * shell_soft, 0.0, 0.72)

    erase_soft = cv2.GaussianBlur(erase.astype(np.float32), (0, 0), sigmaX=2.2) / 255.0
    erase_soft = np.clip(erase_soft, 0.0, 1.0)
    low_trend = clean + 0.12 * (low_original - low_clean)
    fill = low_trend + keep[:, :, None] * high_texture
    blended = original_f * (1.0 - erase_soft[:, :, None]) + fill * erase_soft[:, :, None]
    return np.clip(blended, 0, 255).astype(np.float32)


def _score_adaptive_counterfactual_canvas(
    original: np.ndarray,
    clean: np.ndarray,
    erase: np.ndarray,
    bbox: tuple[int, int, int, int],
    family: str,
    *,
    candidate_index: int,
    sensitivity: float,
    core_texture_keep: float,
    shell_texture_keep: float,
) -> DRFTCanvasQuality:
    x0, y0, x1, y1 = bbox
    gray_orig = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_clean = cv2.cvtColor(np.clip(clean, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    high_orig = gray_orig - cv2.GaussianBlur(gray_orig, (0, 0), sigmaX=4.0)
    high_clean = gray_clean - cv2.GaussianBlur(gray_clean, (0, 0), sigmaX=4.0)
    core = erase > 8
    if not core.any():
        return DRFTCanvasQuality(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, candidate_index, sensitivity, core_texture_keep, shell_texture_keep)

    evidence_before = float(np.mean(np.abs(high_orig[core])))
    evidence_after = float(np.mean(np.abs(high_clean[core])))
    defect_evidence_drop = np.clip((evidence_before - evidence_after) / max(1.0, evidence_before), 0.0, 1.0)

    outside = np.ones(erase.shape, dtype=bool)
    outside[y0:y1, x0:x1] = erase[y0:y1, x0:x1] <= 4
    outside_delta = float(np.mean(np.abs(gray_clean[outside] - gray_orig[outside])) / 255.0) if outside.any() else 0.0
    background_preservation = math.exp(-outside_delta / 0.010)

    ring = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
    ring = np.logical_and(ring, np.logical_not(core))
    if ring.any():
        std_orig = float(np.std(high_orig[ring]))
        std_clean = float(np.std(high_clean[ring]))
        texture_consistency = math.exp(-abs(std_orig - std_clean) / max(2.0, std_orig))
        lap_orig = cv2.Laplacian(gray_orig, cv2.CV_32F)
        lap_clean = cv2.Laplacian(gray_clean, cv2.CV_32F)
        artifact_delta = float(np.mean(np.abs(lap_clean[ring] - lap_orig[ring])) / 255.0)
    else:
        texture_consistency = 1.0
        artifact_delta = 0.0
    artifact_score = math.exp(-artifact_delta / 0.020)

    coverage = float(core[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0
    min_cov, max_cov = _adaptive_coverage_limits(family)
    coverage_penalty = 1.0
    if coverage < min_cov:
        coverage_penalty = max(0.35, coverage / max(min_cov, 1e-6))
    elif coverage > max_cov:
        coverage_penalty = max(0.35, max_cov / max(coverage, 1e-6))

    total = coverage_penalty * (
        0.44 * float(defect_evidence_drop)
        + 0.26 * float(background_preservation)
        + 0.18 * float(texture_consistency)
        + 0.12 * float(artifact_score)
    )
    return DRFTCanvasQuality(
        total=float(total),
        defect_evidence_drop=float(defect_evidence_drop),
        background_preservation=float(background_preservation),
        texture_consistency=float(texture_consistency),
        artifact_score=float(artifact_score),
        erase_coverage=float(coverage),
        candidate_index=int(candidate_index),
        sensitivity=float(sensitivity),
        core_texture_keep=float(core_texture_keep),
        shell_texture_keep=float(shell_texture_keep),
    )


def _class_aware_existing_defect_mask(
    crop: np.ndarray,
    cls_name: str,
    *,
    seed: int = 0,
    sensitivity: float = 1.0,
) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if gray.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    sensitivity = max(0.55, min(1.65, float(sensitivity)))
    base = _weak_existing_defect_mask(crop, seed=seed).astype(bool)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(2.0, min(gray.shape) / 18.0))
    high = gray - low
    z = _robust_z(high)
    dark = z < (-1.55 / sensitivity)
    bright = z > (1.90 / sensitivity)
    edges = cv2.Canny(np.clip(gray, 0, 255).astype(np.uint8), 28, 88) > 0
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    family = family_from_class(cls_name)

    if family in {"scratch", "crack"}:
        line = _line_response(gray)
        pct = max(78.0, 92.0 - 7.5 * sensitivity)
        line_mask = line >= np.percentile(line, pct)
        if family == "scratch":
            mask = base | (line_mask & (bright | edges | (np.abs(z) > 1.30 / sensitivity)))
        else:
            mask = base | (line_mask & (dark | edges | (np.abs(z) > 1.20 / sensitivity)))
        return _filter_components(mask, min_area=3, max_area_fraction=0.60).astype(np.uint8)

    if family in {"pitted", "inclusion"}:
        kernel_size = 9 if min(gray.shape) >= 96 else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(np.clip(gray, 0, 255).astype(np.uint8), cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        tophat = cv2.morphologyEx(np.clip(gray, 0, 255).astype(np.uint8), cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        black_thr = max(4.0, float(np.percentile(blackhat, max(78.0, 90.0 - 7.0 * sensitivity))))
        top_thr = max(4.0, float(np.percentile(tophat, max(84.0, 94.0 - 6.0 * sensitivity))))
        blob = blackhat >= black_thr
        if family == "pitted":
            blob = blob | ((tophat >= top_thr) & edges)
        mask = base | blob | (dark & (edges | (blackhat > 2.0)))
        return _filter_components(mask, min_area=4, max_area_fraction=0.58).astype(np.uint8)

    low_freq = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(4.0, min(gray.shape) / 8.0))
    regional = _robust_z(gray - low_freq)
    region_mask = np.abs(regional) > (1.10 / sensitivity)
    texture = np.abs(high) > np.percentile(np.abs(high), max(74.0, 88.0 - 8.0 * sensitivity))
    mask = base | (region_mask & (texture | edges))
    return _filter_components(mask, min_area=8, max_area_fraction=0.68).astype(np.uint8)


def _line_response(gray: np.ndarray) -> np.ndarray:
    centered = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    response = np.zeros_like(centered, dtype=np.float32)
    for theta in np.linspace(0.0, math.pi, 8, endpoint=False):
        kernel = cv2.getGaborKernel((15, 15), 3.0, float(theta), 8.5, 0.45, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(centered, cv2.CV_32F, kernel)
        response = np.maximum(response, np.abs(filtered))
    scale = max(1.0, float(np.percentile(response, 99)))
    return np.clip(response / scale, 0.0, 1.0)


def _robust_z(arr: np.ndarray) -> np.ndarray:
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = max(2.0, 1.4826 * mad)
    return (arr - median) / scale


def _filter_components(mask: np.ndarray, *, min_area: int, max_area_fraction: float) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return mask_u8
    max_area = max(1, int(mask_u8.size * float(max_area_fraction)))
    keep = np.zeros_like(mask_u8)
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            keep[labels == idx] = 1
    return keep


def _trim_mask_to_coverage(mask: np.ndarray, max_coverage: float) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(np.uint8)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return mask.astype(np.uint8)
    target = max(1, int(mask.size * max_coverage))
    if coords.shape[0] <= target:
        return mask.astype(np.uint8)
    # Prefer compact central evidence when a class-aware detector overfires.
    ys, xs = coords[:, 0].astype(np.float32), coords[:, 1].astype(np.float32)
    cy, cx = float(ys.mean()), float(xs.mean())
    dist = (ys - cy) ** 2 + (xs - cx) ** 2
    keep_idx = np.argsort(dist)[:target]
    trimmed = np.zeros_like(mask, dtype=np.uint8)
    kept = coords[keep_idx]
    trimmed[kept[:, 0], kept[:, 1]] = 1
    return trimmed


def _weak_existing_defect_mask(crop: np.ndarray, *, seed: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if gray.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(2.0, min(gray.shape) / 18.0))
    high = gray - low
    median = float(np.median(high))
    mad = float(np.median(np.abs(high - median)))
    scale = max(2.0, 1.4826 * mad)
    z = (high - median) / scale
    dark = z < -1.85
    bright = z > 2.25

    edges = cv2.Canny(np.clip(gray, 0, 255).astype(np.uint8), 34, 92) > 0
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    mask = np.logical_or(dark, np.logical_and(bright, edges))

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    if rng.random() < 0.35:
        # Keep a fraction of weak candidates so the background branch remains
        # conservative on very textured steel surfaces.
        texture = np.abs(high) / max(1.0, float(np.percentile(np.abs(high), 90)))
        mask = np.logical_and(mask, texture > 0.42)
    return mask.astype(np.uint8)


def _mask_to_float(mask: Image.Image | None, target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    if mask is None:
        return np.zeros((height, width), dtype=np.float32)
    resized = mask.convert("L").resize(target_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _resize_float_field(field: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    arr = np.asarray(field, dtype=np.float32)
    if arr.shape == (height, width):
        return np.clip(arr, 0.0, 1.0)
    return np.clip(cv2.resize(arr, target_size, interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


def _resize_numeric_field(field: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    arr = np.asarray(field, dtype=np.float32)
    if arr.shape == (height, width):
        return arr
    return cv2.resize(arr, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _float_mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")


def _family_seed_amplitude(family: str) -> float:
    family_lc = family.lower().strip()
    if family_lc in {"crack", "crazing"}:
        return 17.0
    if family_lc == "pitted":
        return 19.0
    if family_lc == "inclusion":
        return 17.0
    if family_lc == "scratch":
        return 15.0
    if family_lc in {"patch", "scale"}:
        return 15.5
    return 14.0


def _bbox_from_mask(mask: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return _clip_bbox((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), width, height)


def _merge_bboxes(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if right[2] <= right[0] or right[3] <= right[1]:
        return left
    if left[2] <= left[0] or left[3] <= left[1]:
        return right
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    dilation: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clip_bbox(bbox, width, height)
    box_w = max(1.0, float(x1 - x0))
    box_h = max(1.0, float(y1 - y0))
    cx = 0.5 * float(x0 + x1)
    cy = 0.5 * float(y0 + y1)
    new_w = box_w * max(1.0, float(dilation))
    new_h = box_h * max(1.0, float(dilation))
    return _clip_bbox(
        (
            int(round(cx - 0.5 * new_w)),
            int(round(cy - 0.5 * new_h)),
            int(round(cx + 0.5 * new_w)),
            int(round(cy + 0.5 * new_h)),
        ),
        width,
        height,
    )


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if mask.any():
        return float(values[mask].mean())
    return 0.0


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width, int(x0)))
    y0 = max(0, min(height, int(y0)))
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _smoothstep(value: float, edge0: float, edge1: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return float(x * x * (3.0 - 2.0 * x))


def _normalize_signed(arr: np.ndarray) -> np.ndarray:
    max_abs = max(1.0, float(np.max(np.abs(arr))))
    return np.clip(arr / (2.0 * max_abs) + 0.5, 0.0, 1.0)
