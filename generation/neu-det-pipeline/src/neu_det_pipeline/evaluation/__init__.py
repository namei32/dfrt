from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


class GenerationMetricsEvaluator:
    """Lightweight image-difference evaluator used by the generation CLI."""

    def __init__(self, device: Optional[str | torch.device] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def evaluate(
        self,
        generated_paths: list[Path],
        init_images: dict[str, Path],
        conditioning: dict[str, dict[str, Path]],
        *,
        bbox_map: Optional[dict[str, tuple[int, int, int, int]]] = None,
        bbox_source_size: tuple[int, int] = (200, 200),
    ) -> dict[str, Any]:
        del conditioning
        rows: list[dict[str, Any]] = []
        full_diffs: list[float] = []
        bbox_diffs: list[float] = []
        for path in generated_paths:
            key = Path(path).stem
            init_path = init_images.get(key)
            if init_path is None:
                continue
            generated = Image.open(path).convert("RGB")
            original = Image.open(init_path).resize(generated.size, Image.Resampling.LANCZOS).convert("RGB")
            gen_arr = np.asarray(generated, dtype=np.float32)
            orig_arr = np.asarray(original, dtype=np.float32)
            diff = np.abs(gen_arr - orig_arr).mean(axis=2) / 255.0
            full_delta = float(diff.mean())
            full_diffs.append(full_delta)
            row: dict[str, Any] = {"target_key": key, "mean_abs_delta": full_delta}
            if bbox_map is not None and key in bbox_map:
                bbox_delta = _bbox_delta(diff, bbox_map[key], bbox_source_size, generated.size)
                bbox_diffs.append(bbox_delta)
                row["bbox_mean_abs_delta"] = bbox_delta
            rows.append(row)
        return {
            "generated_count": len(generated_paths),
            "evaluated_count": len(rows),
            "mean_abs_delta": _safe_mean(full_diffs),
            "bbox_mean_abs_delta": _safe_mean(bbox_diffs),
            "rows": rows,
        }


def save_metrics(result: dict[str, Any], path: Path, *, extra: Optional[dict[str, Any]] = None) -> None:
    payload = dict(result)
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_mean(values: list[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _bbox_delta(
    diff: np.ndarray,
    bbox: tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> float:
    src_w, src_h = source_size
    tgt_w, tgt_h = target_size
    x0, y0, x1, y1 = bbox
    tx0 = int(round(float(x0) * tgt_w / max(1.0, float(src_w))))
    tx1 = int(round(float(x1) * tgt_w / max(1.0, float(src_w))))
    ty0 = int(round(float(y0) * tgt_h / max(1.0, float(src_h))))
    ty1 = int(round(float(y1) * tgt_h / max(1.0, float(src_h))))
    tx0, tx1 = sorted((max(0, min(tgt_w, tx0)), max(0, min(tgt_w, tx1))))
    ty0, ty1 = sorted((max(0, min(tgt_h, ty0)), max(0, min(tgt_h, ty1))))
    if tx1 <= tx0 or ty1 <= ty0:
        return 0.0
    return float(diff[ty0:ty1, tx0:tx1].mean())
