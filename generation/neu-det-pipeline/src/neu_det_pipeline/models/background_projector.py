from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data.loader import DefectSample
from ..guidance.drft import build_adaptive_counterfactual_canvas
from ..guidance.mask import create_composite_preserving_background
from ..guidance.morphology import scale_bbox


@dataclass
class BackgroundProjectorConfig:
    """Training and inference knobs for the box-aware background projector."""

    resolution: int = 512
    base_channels: int = 32
    learning_rate: float = 1e-3
    steps: int = 500
    batch_size: int = 4
    seed: int = 42
    box_weight: float = 2.0
    erase_weight: float = 5.0
    context_weight: float = 1.5
    outside_weight: float = 0.25
    gradient_weight: float = 0.08
    uncertainty_weight: float = 0.04
    protect_weight: float = 0.03
    fallback_blend: bool = True


@dataclass
class BackgroundProjectionResult:
    background: Image.Image
    uncertainty: Image.Image
    protect_mask: Image.Image
    meta: dict[str, Any]


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(max(1, min(8, out_channels // 4)), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, out_channels // 4)), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BoxAwareBackgroundProjector(nn.Module):
    """Small U-Net that predicts a pseudo-normal background inside the source box."""

    input_channels: int = 8

    def __init__(self, *, base_channels: int = 32) -> None:
        super().__init__()
        ch = max(8, int(base_channels))
        self.enc1 = _ConvBlock(self.input_channels, ch)
        self.enc2 = _ConvBlock(ch, ch * 2, stride=2)
        self.enc3 = _ConvBlock(ch * 2, ch * 4, stride=2)
        self.mid = _ConvBlock(ch * 4, ch * 4)
        self.dec2 = _ConvBlock(ch * 6, ch * 2)
        self.dec1 = _ConvBlock(ch * 3, ch)
        self.out = nn.Conv2d(ch, 5, kernel_size=1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        source = torch.clamp(features[:, 0:3], -1.0, 1.0)
        box_gate = torch.clamp(features[:, 3:4], 0.0, 1.0)
        erase_gate = torch.clamp(features[:, 4:5], 0.0, 1.0)
        edit_gate = torch.clamp(0.35 * box_gate + 0.65 * erase_gate, 0.0, 1.0)

        e1 = self.enc1(features)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        mid = self.mid(e3)
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        raw = self.out(d1)

        delta = 0.55 * torch.tanh(raw[:, 0:3])
        background = torch.clamp(source + delta * edit_gate, -1.0, 1.0)
        return {
            "background": background,
            "uncertainty": torch.sigmoid(raw[:, 3:4]),
            "protect": torch.sigmoid(raw[:, 4:5]),
        }


class BackgroundProjectionDataset(Dataset[dict[str, torch.Tensor]]):
    """Build weakly supervised projector samples from DRFT counterfactual canvases."""

    def __init__(
        self,
        samples: Sequence[DefectSample],
        *,
        resolution: int,
        canvas_candidates: int = 5,
    ) -> None:
        self.samples = list(samples)
        self.resolution = max(64, int(resolution))
        self.canvas_candidates = max(1, int(canvas_candidates))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        source = Image.open(sample.image_path).convert("RGB")
        original_size = source.size
        source_resized = source.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        scaled_bbox = scale_bbox(sample.bbox, original_size, source_resized.size)
        clean_canvas, erase_mask, _meta = build_adaptive_counterfactual_canvas(
            source_resized,
            scaled_bbox,
            sample.cls_name,
            seed=index + 17,
            candidates=self.canvas_candidates,
        )
        target = create_composite_preserving_background(source_resized, clean_canvas, erase_mask)
        features, masks = build_background_projector_features(
            source_resized,
            scaled_bbox,
            erase_mask=erase_mask,
            resolution=self.resolution,
        )
        return {
            "features": features,
            "source": _image_to_tensor(source_resized),
            "target": _image_to_tensor(target),
            **masks,
        }


class BackgroundProjectorTrainer:
    """Train the projector to replace heuristic background construction at inference."""

    def __init__(self, cfg: BackgroundProjectorConfig) -> None:
        self.cfg = cfg

    def train(
        self,
        samples: Sequence[DefectSample],
        output_dir: Path,
        *,
        max_train_samples: int | None = None,
    ) -> Path:
        if max_train_samples is not None:
            samples = list(samples)[: max(1, int(max_train_samples))]
        if not samples:
            raise ValueError("Background projector training requires at least one sample.")

        torch.manual_seed(int(self.cfg.seed))
        np.random.seed(int(self.cfg.seed))
        output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BoxAwareBackgroundProjector(base_channels=self.cfg.base_channels).to(device)
        dataset = BackgroundProjectionDataset(samples, resolution=self.cfg.resolution)
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(self.cfg.batch_size)),
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.cfg.learning_rate))
        metrics: list[dict[str, float]] = []
        global_step = 0
        model.train()

        while global_step < int(self.cfg.steps):
            for batch in loader:
                features = batch["features"].to(device)
                source = batch["source"].to(device)
                target = batch["target"].to(device)
                box = batch["box_mask"].to(device)
                erase = batch["erase_mask"].to(device)
                context = batch["context_mask"].to(device)

                output = model(features)
                pred = output["background"]
                weight = (
                    float(self.cfg.outside_weight)
                    + float(self.cfg.box_weight) * box
                    + float(self.cfg.erase_weight) * erase
                    + float(self.cfg.context_weight) * context
                )
                rec_loss = (_abs_rgb(pred - target) * weight).mean() / torch.clamp(weight.mean(), min=1e-4)
                outside = 1.0 - torch.clamp(box, 0.0, 1.0)
                identity_loss = (_abs_rgb(pred - source) * outside).mean()
                gradient_loss = _gradient_l1(pred, target, torch.clamp(erase + context, 0.0, 1.0))
                err_target = torch.clamp(_abs_rgb(pred.detach() - target), 0.0, 1.0)
                uncertainty_loss = F.l1_loss(output["uncertainty"], err_target)
                protect_target = _edge_protect_target(source, erase)
                protect_loss = F.l1_loss(output["protect"], protect_target)
                loss = (
                    rec_loss
                    + 0.35 * identity_loss
                    + float(self.cfg.gradient_weight) * gradient_loss
                    + float(self.cfg.uncertainty_weight) * uncertainty_loss
                    + float(self.cfg.protect_weight) * protect_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                metrics.append(
                    {
                        "step": float(global_step),
                        "loss": float(loss.detach().cpu()),
                        "reconstruction": float(rec_loss.detach().cpu()),
                        "identity": float(identity_loss.detach().cpu()),
                    }
                )
                global_step += 1
                if global_step >= int(self.cfg.steps):
                    break

        checkpoint_path = output_dir / "background_projector.pt"
        payload = {
            "variant": "box-aware-counterfactual-background-projector",
            "config": asdict(self.cfg),
            "class_names": sorted({sample.cls_name for sample in samples}),
            "state_dict": _detach_cpu_state_dict(model),
            "metrics": metrics,
        }
        torch.save(payload, checkpoint_path)
        (output_dir / "background_projector_metadata.json").write_text(
            json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2),
            encoding="utf-8",
        )
        return checkpoint_path


def train_background_projector(
    samples: Sequence[DefectSample],
    output_dir: Path,
    *,
    config: BackgroundProjectorConfig | None = None,
    max_train_samples: int | None = None,
) -> Path:
    return BackgroundProjectorTrainer(config or BackgroundProjectorConfig()).train(
        samples,
        output_dir,
        max_train_samples=max_train_samples,
    )


def load_background_projector(
    checkpoint_path: Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[BoxAwareBackgroundProjector, BackgroundProjectorConfig, dict[str, Any]]:
    device_obj = torch.device(device or "cpu")
    payload = torch.load(checkpoint_path, map_location=device_obj)
    cfg = _config_from_payload(payload.get("config") or {})
    model = BoxAwareBackgroundProjector(base_channels=cfg.base_channels)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.to(device_obj)
    model.eval()
    return model, cfg, payload


def apply_background_projector(
    model: BoxAwareBackgroundProjector,
    cfg: BackgroundProjectorConfig,
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    erase_mask: Image.Image | None = None,
    fallback_canvas: Image.Image | None = None,
    checkpoint_path: Path | None = None,
    device: torch.device | str | None = None,
) -> BackgroundProjectionResult:
    original_size = image.size
    resolution = max(64, int(cfg.resolution))
    work_image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    scaled_bbox = scale_bbox(bbox, original_size, work_image.size)
    work_erase = None
    if erase_mask is not None:
        work_erase = erase_mask.resize(work_image.size, Image.Resampling.BILINEAR)
    features, _masks = build_background_projector_features(
        work_image,
        scaled_bbox,
        erase_mask=work_erase,
        resolution=resolution,
    )
    device_obj = next(model.parameters()).device if device is None else torch.device(device)
    with torch.inference_mode():
        output = model(features.unsqueeze(0).to(device_obj))
    learned = _tensor_to_image(output["background"][0].cpu()).resize(original_size, Image.Resampling.LANCZOS)
    uncertainty = _mask_tensor_to_image(output["uncertainty"][0].cpu()).resize(
        original_size,
        Image.Resampling.BILINEAR,
    )
    protect = _mask_tensor_to_image(output["protect"][0].cpu()).resize(
        original_size,
        Image.Resampling.BILINEAR,
    )
    fallback = (fallback_canvas or image).convert("RGB").resize(original_size, Image.Resampling.LANCZOS)
    background = _compose_projected_background(
        image.convert("RGB"),
        learned,
        fallback,
        uncertainty,
        bbox,
        fallback_blend=bool(cfg.fallback_blend),
    )
    uncertainty_arr = np.asarray(uncertainty.convert("L"), dtype=np.float32) / 255.0
    protect_arr = np.asarray(protect.convert("L"), dtype=np.float32) / 255.0
    meta = {
        "enabled": True,
        "variant": "box-aware-counterfactual-background-projector",
        "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None,
        "resolution": int(resolution),
        "fallback_blend": bool(cfg.fallback_blend),
        "uncertainty_mean": float(uncertainty_arr.mean()),
        "uncertainty_p90": float(np.percentile(uncertainty_arr, 90)),
        "protect_mean": float(protect_arr.mean()),
    }
    return BackgroundProjectionResult(background=background, uncertainty=uncertainty, protect_mask=protect, meta=meta)


def build_background_projector_features(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    erase_mask: Image.Image | None = None,
    resolution: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    work_size = (int(resolution), int(resolution)) if resolution is not None else image.size
    source = image.convert("RGB").resize(work_size, Image.Resampling.LANCZOS)
    box_mask = _box_mask(work_size, bbox if source.size == image.size else scale_bbox(bbox, image.size, work_size))
    if erase_mask is None:
        erase = box_mask
    else:
        erase = erase_mask.convert("L").resize(work_size, Image.Resampling.BILINEAR)
    context = _context_mask(work_size, bbox if source.size == image.size else scale_bbox(bbox, image.size, work_size))
    coords = _coord_tensor(work_size, bbox if source.size == image.size else scale_bbox(bbox, image.size, work_size))
    masks = {
        "box_mask": _mask_to_tensor(box_mask),
        "erase_mask": _mask_to_tensor(erase),
        "context_mask": _mask_to_tensor(context),
    }
    features = torch.cat(
        [
            _image_to_tensor(source),
            masks["box_mask"],
            masks["erase_mask"],
            masks["context_mask"],
            coords,
        ],
        dim=0,
    )
    return features.contiguous(), masks


def _config_from_payload(raw: dict[str, Any]) -> BackgroundProjectorConfig:
    valid = {field.name for field in fields(BackgroundProjectorConfig)}
    kwargs = {key: value for key, value in raw.items() if key in valid}
    return BackgroundProjectorConfig(**kwargs)


def _box_mask(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> Image.Image:
    width, height = size
    x0, y0, x1, y1 = _clip_bbox(bbox, size)
    mask = Image.new("L", size, 0)
    if x1 > x0 and y1 > y0:
        ImageDraw.Draw(mask).rectangle((x0, y0, x1, y1), fill=255)
    return mask


def _context_mask(size: tuple[int, int], bbox: tuple[int, int, int, int], *, scale: float = 1.45) -> Image.Image:
    width, height = size
    x0, y0, x1, y1 = _clip_bbox(bbox, size)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = max(1.0, (x1 - x0) * float(scale))
    bh = max(1.0, (y1 - y0) * float(scale))
    expanded = (
        int(max(0, round(cx - bw * 0.5))),
        int(max(0, round(cy - bh * 0.5))),
        int(min(width, round(cx + bw * 0.5))),
        int(min(height, round(cy + bh * 0.5))),
    )
    outer = np.asarray(_box_mask(size, expanded), dtype=np.uint8)
    inner = np.asarray(_box_mask(size, (x0, y0, x1, y1)), dtype=np.uint8)
    ring = np.clip(outer.astype(np.int16) - inner.astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(ring, mode="L")


def _coord_tensor(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> torch.Tensor:
    width, height = size
    x0, y0, x1, y1 = _clip_bbox(bbox, size)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = max(1.0, float(x1 - x0))
    bh = max(1.0, float(y1 - y0))
    yy, xx = np.mgrid[:height, :width]
    x_rel = np.clip((xx.astype(np.float32) - cx) / bw, -1.5, 1.5) / 1.5
    y_rel = np.clip((yy.astype(np.float32) - cy) / bh, -1.5, 1.5) / 1.5
    return torch.from_numpy(np.stack([x_rel, y_rel], axis=0)).float().contiguous()


def _clip_bbox(bbox: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
    y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
    return x0, y0, x1, y1


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).contiguous()


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(-1.0, 1.0).permute(1, 2, 0).numpy()
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr[None]).contiguous()


def _mask_tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(0.0, 1.0).squeeze(0).numpy()
    return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L")


def _abs_rgb(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.abs().mean(dim=1, keepdim=True)


def _gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_x = mask[:, :, :, 1:]
    mask_y = mask[:, :, 1:, :]
    pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    targ_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    targ_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (torch.abs(pred_x - targ_x) * mask_x).mean() + (torch.abs(pred_y - targ_y) * mask_y).mean()


def _edge_protect_target(source: torch.Tensor, erase: torch.Tensor) -> torch.Tensor:
    gray = source.mean(dim=1, keepdim=True)
    dx = F.pad(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), (0, 0, 0, 1))
    edge = torch.clamp((dx + dy) * 2.0, 0.0, 1.0)
    return edge * (1.0 - torch.clamp(erase, 0.0, 1.0))


def _compose_projected_background(
    source: Image.Image,
    learned: Image.Image,
    fallback: Image.Image,
    uncertainty: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    fallback_blend: bool,
) -> Image.Image:
    source_arr = np.asarray(source.convert("RGB"), dtype=np.float32)
    learned_arr = np.asarray(learned.convert("RGB"), dtype=np.float32)
    fallback_arr = np.asarray(fallback.convert("RGB"), dtype=np.float32)
    box = np.asarray(_box_mask(source.size, bbox), dtype=np.float32) / 255.0
    if fallback_blend:
        confidence = 1.0 - np.asarray(uncertainty.convert("L"), dtype=np.float32) / 255.0
        inside = learned_arr * confidence[..., None] + fallback_arr * (1.0 - confidence[..., None])
    else:
        inside = learned_arr
    result = source_arr * (1.0 - box[..., None]) + inside * box[..., None]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def _detach_cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}
