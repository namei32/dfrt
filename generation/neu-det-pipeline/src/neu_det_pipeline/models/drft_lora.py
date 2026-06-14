from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from diffusers.optimization import get_scheduler
from diffusers.schedulers import DDPMScheduler
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config import LoRAConfig
from ..data.loader import DefectSample
from ..guidance.drft import (
    DRFTResidualField,
    build_adaptive_counterfactual_canvas,
    build_class_aware_defect_residual_field,
)
from ..guidance.morphology import MorphologyPrior, build_morphology_calibrated_mask, scale_bbox


DRFT_ADAPTER_TYPE = "drft_lora"
DRFT_FIELD_CHANNELS = 6
DRFT_FIELD_STATS = DRFT_FIELD_CHANNELS * 2


@dataclass
class DRFTLoRAHyperParams:
    rank: int = 4
    alpha: int = 4
    num_experts: int = 3
    max_gate: float = 1.25


def is_drft_lora_path(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        return metadata.get("adapter_type") == DRFT_ADAPTER_TYPE
    except Exception:
        return False


def read_drft_lora_metadata(path: Path) -> Dict[str, Any]:
    path = Path(path)
    metadata: Dict[str, Any] = {}
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw = handle.metadata() or {}
        metadata.update(raw)
    except Exception:
        pass
    config_path = path.with_name("drft_lora_config.json")
    if config_path.exists():
        try:
            metadata.update(json.loads(config_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    class_names = metadata.get("class_names")
    if isinstance(class_names, str):
        try:
            metadata["class_names"] = json.loads(class_names)
        except json.JSONDecodeError:
            metadata["class_names"] = [name for name in class_names.split(",") if name]
    return metadata


def residual_field_to_tensor(field: DRFTResidualField) -> torch.Tensor:
    signed = np.clip(field.signed_residual.astype(np.float32) / 40.0, -1.0, 1.0)
    data = np.stack(
        [
            np.clip(field.soft_mask, 0.0, 1.0),
            signed,
            np.clip(field.orientation_x, -1.0, 1.0),
            np.clip(field.orientation_y, -1.0, 1.0),
            np.clip(field.boundary_shell, 0.0, 1.0),
            np.clip(field.distance, 0.0, 1.0),
        ],
        axis=0,
    )
    return torch.from_numpy(data.astype(np.float32))


def build_observed_residual_field(
    defect_image: Image.Image,
    clean_canvas: Image.Image,
    defect_mask: Image.Image,
    cls_name: str,
    *,
    orientation_deg: float,
    seed: int,
) -> DRFTResidualField:
    field = build_class_aware_defect_residual_field(
        clean_canvas,
        defect_mask,
        cls_name,
        orientation_deg=orientation_deg,
        seed=seed,
    )
    defect = np.asarray(defect_image.convert("L"), dtype=np.float32)
    clean = np.asarray(clean_canvas.convert("L"), dtype=np.float32)
    if defect.shape != field.soft_mask.shape:
        defect = cv2.resize(defect, field.soft_mask.shape[::-1], interpolation=cv2.INTER_AREA)
        clean = cv2.resize(clean, field.soft_mask.shape[::-1], interpolation=cv2.INTER_AREA)
    observed = (defect - clean) * np.clip(field.soft_mask, 0.0, 1.0)
    observed = cv2.GaussianBlur(observed.astype(np.float32), (0, 0), sigmaX=0.65)
    core = field.soft_mask > 0.18
    if core.any():
        mean_abs = float(np.mean(np.abs(observed[core])))
        if mean_abs > 1.0:
            observed = np.clip(observed * min(2.0, 18.0 / mean_abs), -38.0, 38.0)
    field.signed_residual = observed.astype(np.float32)
    return field


class DRFTAttentionContext:
    """Runtime residual-field context shared by all DRFT attention processors."""

    def __init__(self, class_count: int, num_train_timesteps: int = 1000) -> None:
        self.class_count = max(1, int(class_count))
        self.num_train_timesteps = max(1, int(num_train_timesteps))
        self.residual_field: Optional[torch.Tensor] = None
        self.class_ids: Optional[torch.Tensor] = None
        self.timestep: Optional[torch.Tensor] = None
        self._cache: Dict[Tuple[str, str, int, torch.dtype, int], torch.Tensor] = {}

    def set_condition(self, residual_field: torch.Tensor, class_ids: torch.Tensor) -> None:
        if residual_field.ndim != 4 or residual_field.shape[1] != DRFT_FIELD_CHANNELS:
            raise ValueError(
                f"residual_field must have shape [B,{DRFT_FIELD_CHANNELS},H,W], got {tuple(residual_field.shape)}"
            )
        self.residual_field = residual_field.detach()
        self.class_ids = class_ids.detach().long()
        self._cache.clear()

    def set_timestep(self, timestep: Any) -> None:
        if isinstance(timestep, torch.Tensor):
            self.timestep = timestep.detach().flatten().float()
        else:
            self.timestep = torch.tensor([float(timestep)], dtype=torch.float32)
        self._cache.clear()

    def clear(self) -> None:
        self.residual_field = None
        self.class_ids = None
        self.timestep = None
        self._cache.clear()

    @staticmethod
    def _repeat_to_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
        if tensor.shape[0] == batch:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.repeat(batch, *([1] * (tensor.ndim - 1)))
        reps = math.ceil(batch / tensor.shape[0])
        return tensor.repeat(reps, *([1] * (tensor.ndim - 1)))[:batch]

    def field_stats(self, *, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.residual_field is None:
            return torch.zeros(batch, DRFT_FIELD_STATS, device=device, dtype=dtype)
        key = ("stats", str(device), batch, dtype, 0)
        if key in self._cache:
            return self._cache[key]
        field = self._repeat_to_batch(self.residual_field.to(device=device, dtype=dtype), batch)
        means = field.flatten(2).mean(dim=-1)
        stds = field.flatten(2).std(dim=-1, unbiased=False)
        stats = torch.cat([means, stds], dim=-1)
        self._cache[key] = stats
        return stats

    def class_tensor(self, *, batch: int, device: torch.device) -> torch.Tensor:
        if self.class_ids is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        class_ids = self._repeat_to_batch(self.class_ids.to(device=device, dtype=torch.long).view(-1, 1), batch).view(-1)
        return torch.clamp(class_ids, 0, self.class_count - 1)

    def time_features(self, *, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.timestep is None:
            t = torch.zeros(batch, device=device, dtype=dtype)
        else:
            t = self._repeat_to_batch(self.timestep.to(device=device, dtype=dtype).view(-1, 1), batch).view(-1)
            t = torch.clamp(t / float(self.num_train_timesteps - 1), 0.0, 1.0)
        early = torch.sigmoid((t - 0.62) * 12.0)
        mid = torch.exp(-((t - 0.45) ** 2) / 0.055)
        late = torch.sigmoid((0.28 - t) * 12.0)
        return torch.stack(
            [
                t,
                1.0 - t,
                early,
                mid,
                late,
                torch.sin(math.pi * t),
                torch.cos(math.pi * t),
                torch.sin(2.0 * math.pi * t),
            ],
            dim=-1,
        )

    def spatial_gate(self, *, seq_len: int, batch: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.residual_field is None or seq_len <= 1:
            return None
        side = int(round(math.sqrt(seq_len)))
        if side * side != seq_len:
            return None
        key = ("spatial", str(device), batch, dtype, seq_len)
        if key in self._cache:
            return self._cache[key]
        field = self._repeat_to_batch(self.residual_field.to(device=device, dtype=dtype), batch)
        mask = torch.clamp(field[:, 0:1] + 0.35 * field[:, 4:5], 0.0, 1.0)
        resized = F.interpolate(mask, size=(side, side), mode="bilinear", align_corners=False)
        gate = resized.flatten(2).transpose(1, 2)
        self._cache[key] = gate
        return gate


class _DRFTLoRALinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: int, num_experts: int) -> None:
        super().__init__()
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.num_experts = int(num_experts)
        self.down = nn.ModuleList([nn.Linear(in_dim, rank, bias=False) for _ in range(num_experts)])
        self.up = nn.ModuleList([nn.Linear(rank, out_dim, bias=False) for _ in range(num_experts)])
        self.scale = float(alpha) / float(rank)
        for down, up in zip(self.down, self.up):
            nn.init.normal_(down.weight, std=1.0 / max(1, rank))
            nn.init.zeros_(up.weight)

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        output = torch.zeros(*x.shape[:-1], self.up[0].out_features, device=x.device, dtype=target_dtype)
        for idx, (down, up) in enumerate(zip(self.down, self.up)):
            if down.weight.dtype != target_dtype:
                down.to(dtype=target_dtype)
                up.to(dtype=target_dtype)
            delta = up(down(x)) * self.scale
            output = output + delta * expert_weights[:, idx].view(-1, 1, 1).to(dtype=target_dtype)
        return output


class DRFTAttnProcessor2_0(nn.Module):
    """Attention processor implementing residual-field/timestep/class-gated LoRA."""

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int],
        *,
        rank: int,
        alpha: int,
        num_experts: int,
        class_count: int,
        context: DRFTAttentionContext,
        max_gate: float = 1.25,
    ) -> None:
        super().__init__()
        cross_attention_dim = cross_attention_dim or hidden_size
        gate_hidden = 64
        self.hidden_size = int(hidden_size)
        self.cross_attention_dim = int(cross_attention_dim)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.num_experts = int(num_experts)
        self.class_count = int(class_count)
        self.max_gate = float(max_gate)
        self.context = context
        self.lora_q = _DRFTLoRALinear(hidden_size, hidden_size, rank, alpha, num_experts)
        self.lora_k = _DRFTLoRALinear(cross_attention_dim, hidden_size, rank, alpha, num_experts)
        self.lora_v = _DRFTLoRALinear(cross_attention_dim, hidden_size, rank, alpha, num_experts)
        self.lora_out = _DRFTLoRALinear(hidden_size, hidden_size, rank, alpha, num_experts)
        self.field_proj = nn.Sequential(nn.Linear(DRFT_FIELD_STATS, gate_hidden), nn.SiLU(), nn.Linear(gate_hidden, gate_hidden))
        self.time_proj = nn.Sequential(nn.Linear(8, gate_hidden), nn.SiLU(), nn.Linear(gate_hidden, gate_hidden))
        self.class_embed = nn.Embedding(max(1, class_count), gate_hidden)
        self.router = nn.Sequential(nn.SiLU(), nn.Linear(gate_hidden, num_experts + 1))

    @staticmethod
    def _reshape_if_needed(hidden_states: torch.Tensor, info: Tuple[int, Optional[int], Optional[int], Optional[int]]) -> torch.Tensor:
        batch_size, channel, height, width = info
        if channel is None:
            return hidden_states
        return hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    def _route(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch = hidden_states.shape[0]
        stats = self.context.field_stats(batch=batch, device=hidden_states.device, dtype=hidden_states.dtype)
        time_features = self.context.time_features(batch=batch, device=hidden_states.device, dtype=hidden_states.dtype)
        class_ids = self.context.class_tensor(batch=batch, device=hidden_states.device)
        gate_hidden = self.field_proj(stats) + self.time_proj(time_features) + self.class_embed(class_ids).to(hidden_states.dtype)
        logits = self.router(gate_hidden)
        expert_weights = torch.softmax(logits[:, : self.num_experts], dim=-1)
        alpha = torch.sigmoid(logits[:, self.num_experts : self.num_experts + 1]) * self.max_gate
        spatial = self.context.spatial_gate(
            seq_len=hidden_states.shape[1],
            batch=batch,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return expert_weights, alpha, spatial

    def _preprocess(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        temb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, Tuple[int, Optional[int], Optional[int], Optional[int]]]:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            target_len = hidden_states.shape[1]
        else:
            target_len = encoder_hidden_states.shape[1]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        attention_mask = attn.prepare_attention_mask(attention_mask, target_len, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        return residual, hidden_states, encoder_hidden_states, attention_mask, input_ndim, (batch_size, channel, height, width)

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        residual, hidden_states, encoder_hidden_states, attention_mask, input_ndim, shape_info = self._preprocess(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )
        expert_weights, alpha, spatial = self._route(hidden_states)
        q_delta = self.lora_q(hidden_states, expert_weights)
        if spatial is not None:
            q_delta = q_delta * (0.20 + 0.80 * spatial)
        query = attn.to_q(hidden_states) + alpha[:, None, :] * q_delta

        key = attn.to_k(encoder_hidden_states) + alpha[:, None, :] * self.lora_k(encoder_hidden_states, expert_weights)
        value = attn.to_v(encoder_hidden_states) + alpha[:, None, :] * self.lora_v(encoder_hidden_states, expert_weights)

        batch_size = hidden_states.shape[0]
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out_delta = self.lora_out(hidden_states, expert_weights)
        if spatial is not None and spatial.shape[1] == out_delta.shape[1]:
            out_delta = out_delta * (0.20 + 0.80 * spatial)
        hidden_states = attn.to_out[0](hidden_states) + alpha[:, None, :] * out_delta
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = self._reshape_if_needed(hidden_states, shape_info)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def _resolve_attention_module(unet: Any, processor_name: str) -> Any:
    module = unet
    for part in processor_name.split(".")[:-1]:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _hidden_size_for_processor(unet: Any, processor_name: str) -> int:
    try:
        module = _resolve_attention_module(unet, processor_name)
        if hasattr(module, "to_q"):
            return int(module.to_q.in_features)
    except Exception:
        pass
    if processor_name.startswith("mid_block"):
        return int(unet.config.block_out_channels[-1])
    if processor_name.startswith("up_blocks"):
        return int(unet.config.block_out_channels[-1])
    return int(unet.config.block_out_channels[0])


def inject_drft_lora_processors(
    unet: Any,
    *,
    class_names: Sequence[str],
    rank: int,
    alpha: int,
    num_experts: int,
    max_gate: float = 1.25,
    context: Optional[DRFTAttentionContext] = None,
) -> DRFTAttentionContext:
    context = context or DRFTAttentionContext(class_count=max(1, len(class_names)))
    processors: Dict[str, nn.Module] = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = unet.config.cross_attention_dim if name.endswith("attn2.processor") else None
        processors[name] = DRFTAttnProcessor2_0(
            _hidden_size_for_processor(unet, name),
            cross_attention_dim,
            rank=rank,
            alpha=alpha,
            num_experts=num_experts,
            class_count=max(1, len(class_names)),
            context=context,
            max_gate=max_gate,
        )
    unet.set_attn_processor(processors)
    return context


def install_drft_timestep_hook(unet: Any, context: DRFTAttentionContext) -> None:
    if getattr(unet, "_drft_hook_installed", False):
        unet._drft_context = context
        return
    original_forward = unet.forward

    def _forward(self: Any, sample: torch.Tensor, timestep: Any, *args: Any, **kwargs: Any) -> Any:
        self._drft_context.set_timestep(timestep)
        return original_forward(sample, timestep, *args, **kwargs)

    unet._drft_context = context
    unet._drft_original_forward = original_forward
    unet.forward = MethodType(_forward, unet)
    unet._drft_hook_installed = True


def drft_lora_state_dict(unet: Any) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, processor in unet.attn_processors.items():
        if not isinstance(processor, DRFTAttnProcessor2_0):
            continue
        safe_name = name.replace(".", "_")
        for key, value in processor.state_dict().items():
            if isinstance(value, torch.Tensor):
                state[f"{safe_name}.{key}"] = value.detach().cpu()
    return state


def load_drft_lora_into_unet(
    unet: Any,
    lora_path: Path,
    *,
    context: Optional[DRFTAttentionContext] = None,
) -> Tuple[DRFTAttentionContext, Dict[str, Any]]:
    metadata = read_drft_lora_metadata(lora_path)
    class_names = metadata.get("class_names") or []
    if isinstance(class_names, str):
        class_names = [name for name in class_names.split(",") if name]
    rank = int(metadata.get("rank", 4))
    alpha = int(metadata.get("alpha", rank))
    num_experts = int(metadata.get("num_experts", 3))
    max_gate = float(metadata.get("max_gate", 1.25))
    context = inject_drft_lora_processors(
        unet,
        class_names=class_names,
        rank=rank,
        alpha=alpha,
        num_experts=num_experts,
        max_gate=max_gate,
        context=context,
    )
    state = load_file(str(lora_path), device="cpu")
    processor_state: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        head, sep, tail = key.partition(".")
        if not sep:
            continue
        processor_state.setdefault(head, {})[tail] = tensor
    for name, processor in unet.attn_processors.items():
        safe_name = name.replace(".", "_")
        if safe_name in processor_state:
            processor.load_state_dict(processor_state[safe_name], strict=False)
    install_drft_timestep_hook(unet, context)
    return context, metadata


class _DRFTProcessorLayers(nn.Module):
    def __init__(self, processors: Dict[str, nn.Module]) -> None:
        super().__init__()
        for idx, (name, processor) in enumerate(processors.items()):
            safe = name.replace(".", "_")
            if hasattr(self, safe):
                safe = f"{safe}_{idx}"
            self.add_module(safe, processor)


@dataclass
class _DRFTBatchSample:
    defect: torch.Tensor
    clean: torch.Tensor
    mask: torch.Tensor
    field: torch.Tensor
    prompt: str
    class_id: int


class DRFTLoRADataset(Dataset[_DRFTBatchSample]):
    def __init__(
        self,
        samples: Sequence[DefectSample],
        *,
        class_to_id: Dict[str, int],
        token_map: Dict[str, str],
        prompt_template: str,
        resolution: int,
    ) -> None:
        self.samples = list(samples)
        self.class_to_id = class_to_id
        self.token_map = token_map
        self.prompt_template = prompt_template
        self.resolution = int(resolution)
        self.prior = MorphologyPrior.from_samples(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _image_to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    @staticmethod
    def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
        arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr[None]).contiguous()

    def __getitem__(self, index: int) -> _DRFTBatchSample:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        original_size = image.size
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        scaled_bbox = scale_bbox(sample.bbox, original_size, image.size)
        mask, plan = build_morphology_calibrated_mask(
            self.prior,
            sample.cls_name,
            sample.bbox,
            target_key=sample.image_path.stem,
            target_size=image.size,
            target_source_size=original_size,
            candidate_index=0,
            seed=index + 17,
            feather_radius=1.4,
        )
        clean, _erase, _canvas_meta = build_adaptive_counterfactual_canvas(
            image,
            scaled_bbox,
            sample.cls_name,
            seed=index + 17,
            candidates=5,
        )
        field = build_observed_residual_field(
            image,
            clean,
            mask,
            sample.cls_name,
            orientation_deg=plan.orientation_deg,
            seed=index + 17,
        )
        token = self.token_map.get(sample.cls_name, sample.cls_name)
        return _DRFTBatchSample(
            defect=self._image_to_tensor(image),
            clean=self._image_to_tensor(clean),
            mask=self._mask_to_tensor(mask),
            field=residual_field_to_tensor(field),
            prompt=self.prompt_template.format(token=token),
            class_id=self.class_to_id.get(sample.cls_name, 0),
        )


def _collate_drft(batch: Sequence[_DRFTBatchSample]) -> Dict[str, Any]:
    return {
        "defect": torch.stack([item.defect for item in batch]),
        "clean": torch.stack([item.clean for item in batch]),
        "mask": torch.stack([item.mask for item in batch]),
        "field": torch.stack([item.field for item in batch]),
        "prompts": [item.prompt for item in batch],
        "class_ids": torch.tensor([item.class_id for item in batch], dtype=torch.long),
    }


class DRFTLoRATrainer:
    """Train Defect Residual Field + timestep-gated LoRA on SD inpainting U-Net."""

    def __init__(self, cfg: LoRAConfig, hparams: DRFTLoRAHyperParams) -> None:
        self.cfg = cfg
        self.hparams = hparams

    @staticmethod
    def _require_cuda() -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DRFT-LoRA training.")
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"Using CUDA device {idx}: {props.name} ({props.total_memory / (1024 ** 3):.1f} GB)")
        return torch.device(f"cuda:{idx}")

    def _prepare_pipeline(self) -> StableDiffusionInpaintPipeline:
        device = self._require_cuda()
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.unet.requires_grad_(False)
        try:
            pipe.unet.enable_gradient_checkpointing()
        except Exception:
            pass
        return pipe

    @staticmethod
    def _encode(pipe: StableDiffusionInpaintPipeline, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
        scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        return latents * scaling

    def train(
        self,
        samples: Sequence[DefectSample],
        token_map: Dict[str, str],
        output_dir: Path,
        *,
        max_train_samples: Optional[int] = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        if max_train_samples is not None:
            samples = list(samples)[: max(1, int(max_train_samples))]
        class_names = sorted({sample.cls_name for sample in samples})
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        pipe = self._prepare_pipeline()
        context = inject_drft_lora_processors(
            pipe.unet,
            class_names=class_names,
            rank=self.hparams.rank,
            alpha=self.hparams.alpha,
            num_experts=self.hparams.num_experts,
            max_gate=self.hparams.max_gate,
        )
        install_drft_timestep_hook(pipe.unet, context)
        layers = _DRFTProcessorLayers(pipe.unet.attn_processors).to(pipe.unet.device)
        layers.requires_grad_(True)
        trainable = [param for param in layers.parameters() if param.requires_grad]
        if not trainable:
            raise RuntimeError("DRFT-LoRA injection produced no trainable parameters.")

        dataset = DRFTLoRADataset(
            samples,
            class_to_id=class_to_id,
            token_map=token_map,
            prompt_template=self.cfg.prompt_template,
            resolution=self.cfg.resolution,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=_collate_drft,
        )
        optimizer = torch.optim.AdamW(trainable, lr=self.cfg.learning_rate)
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
        )
        lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps,
            num_training_steps=self.cfg.steps,
        )
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        device = pipe.unet.device
        metrics = {"steps": [], "loss": [], "learning_rate": []}
        progress = tqdm(range(self.cfg.steps), desc="DRFT-LoRA training", leave=False)
        optimizer.zero_grad()
        global_step = 0
        running_loss = 0.0
        grad_accum = max(1, int(self.cfg.gradient_accumulation_steps))

        while global_step < self.cfg.steps:
            for batch in loader:
                prompts = batch["prompts"]
                input_ids = tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                    defect_pixels = batch["defect"].to(device)
                    clean_pixels = batch["clean"].to(device)
                    image_mask = batch["mask"].to(device=device, dtype=clean_pixels.dtype)
                    masked_clean_pixels = clean_pixels * (1.0 - image_mask)
                    defect_latents = self._encode(pipe, defect_pixels)
                    clean_latents = self._encode(pipe, masked_clean_pixels)
                noise = torch.randn_like(defect_latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (defect_latents.shape[0],),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(defect_latents, noise, timesteps)
                mask = batch["mask"].to(device=device, dtype=defect_latents.dtype)
                mask_latents = F.interpolate(mask, size=defect_latents.shape[-2:], mode="nearest")
                model_input = torch.cat([noisy_latents, mask_latents, clean_latents], dim=1)
                field_condition = batch["field"].to(device=device, dtype=defect_latents.dtype)
                context.set_condition(field_condition, batch["class_ids"].to(device=device))
                model_pred = pipe.unet(model_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                core = F.interpolate(
                    torch.clamp(field_condition[:, 0:1], 0.0, 1.0),
                    size=model_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                shell = F.interpolate(
                    torch.clamp(field_condition[:, 4:5], 0.0, 1.0),
                    size=model_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                spatial_weight = 1.0 + 5.0 * core + 2.5 * shell
                loss_map = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                loss = (loss_map * spatial_weight.float()).mean() / torch.clamp(spatial_weight.mean(), min=1.0)
                (loss / grad_accum).backward()
                running_loss += float(loss.detach().cpu())
                if (global_step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, self.cfg.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    avg_loss = running_loss / float(grad_accum)
                    metrics["steps"].append(global_step)
                    metrics["loss"].append(avg_loss)
                    metrics["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
                    progress.set_postfix({"loss": f"{avg_loss:.4f}"})
                    running_loss = 0.0
                progress.update(1)
                global_step += 1
                if global_step >= self.cfg.steps:
                    break

        output_path = output_dir / "drft_lora.safetensors"
        metadata = {
            "adapter_type": DRFT_ADAPTER_TYPE,
            "model_id": self.cfg.model_id,
            "rank": str(self.hparams.rank),
            "alpha": str(self.hparams.alpha),
            "num_experts": str(self.hparams.num_experts),
            "max_gate": str(self.hparams.max_gate),
            "class_names": json.dumps(class_names, ensure_ascii=False),
        }
        save_file(drft_lora_state_dict(pipe.unet), str(output_path), metadata=metadata)
        config = {
            "adapter_type": DRFT_ADAPTER_TYPE,
            "model_id": self.cfg.model_id,
            "rank": self.hparams.rank,
            "alpha": self.hparams.alpha,
            "num_experts": self.hparams.num_experts,
            "max_gate": self.hparams.max_gate,
            "class_names": class_names,
            "class_to_id": class_to_id,
            "field_channels": DRFT_FIELD_CHANNELS,
            "training_hyperparameters": {
                "steps": self.cfg.steps,
                "batch_size": self.cfg.batch_size,
                "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
                "learning_rate": self.cfg.learning_rate,
                "resolution": self.cfg.resolution,
                "lr_scheduler": self.cfg.lr_scheduler,
                "lr_warmup_steps": self.cfg.lr_warmup_steps,
                "counterfactual_canvas": "adaptive_drft_v2",
                "loss": "mask_weighted_noise_mse",
            },
        }
        (output_dir / "drft_lora_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "drft_lora_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        progress.close()
        try:
            pipe.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output_path
