from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = ROOT / "generation" / "neu-det-pipeline" / "src"
sys.path.insert(0, str(PIPELINE_SRC))

from neu_det_pipeline.data.resplit import create_mixed_dataset  # noqa: E402


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _clean_target(path: Path) -> None:
    path = path.resolve()
    allowed = (ROOT / "data" / "NEU" / "formal_lora_native").resolve()
    if not _is_relative_to(path, allowed):
        raise RuntimeError(f"Refusing to clean outside NEU formal_lora_native: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _count_images(path: Path, split: str) -> int:
    image_dir = path / "images" / split
    if not image_dir.exists():
        return 0
    return sum(1 for item in image_dir.iterdir() if item.is_file())


def _count_generated(path: Path, split: str) -> int:
    image_dir = path / "images" / split
    if not image_dir.exists():
        return 0
    return sum(1 for item in image_dir.iterdir() if item.is_file() and "_gen" in item.stem)


def _count_boxes(path: Path, split: str) -> int:
    label_dir = path / "labels" / split
    if not label_dir.exists():
        return 0
    boxes = 0
    for label in label_dir.glob("*.txt"):
        for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                boxes += 1
    return boxes


def _make_relative_data_yaml(path: Path) -> None:
    data_yaml = path / "data.yaml"
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    payload["path"] = "."
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _summarize_dataset(path: Path, ratio: int, seed: int, generated_dir: Path | None) -> dict[str, Any]:
    row = {
        "dataset": "neu",
        "label": "NEU-DET",
        "ratio_percent": ratio,
        "seed": seed,
        "root": str(path.resolve()),
        "generated_dir": str(generated_dir.resolve()) if generated_dir else None,
        "splits": {},
        "leakage_audit": {
            "generated_train": _count_generated(path, "train"),
            "generated_val": _count_generated(path, "val"),
            "generated_test": _count_generated(path, "test"),
        },
        "data_yaml": str((path / "data.yaml").resolve()),
    }
    for split in ("train", "val", "test"):
        row["splits"][split] = {
            "images": _count_images(path, split),
            "labels": len(list((path / "labels" / split).glob("*.txt"))) if (path / "labels" / split).exists() else 0,
            "boxes": _count_boxes(path, split),
            "generated": _count_generated(path, split),
        }
    return row


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = (ROOT / "data" / "NEU" / "formal_lora_native").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    empty_generated = out_root / "_empty_generated"
    empty_generated.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for ratio in (0, 100):
        target = out_root / f"neu_drft_lora_native_ratio_{ratio}_dataset"
        _clean_target(target)
        generated_dir = None if ratio == 0 else args.generated_images_dir
        create_mixed_dataset(
            orig_images_dir=args.orig_images_dir,
            orig_labels_dir=args.orig_labels_dir,
            manifest_path=args.split_manifest,
            new_images_dir=generated_dir or empty_generated,
            run_output_dir=target,
            seed=args.seed,
            score_path=args.scores_path if ratio == 100 else None,
            selection_strategy="quality",
        )
        _make_relative_data_yaml(target)
        row = _summarize_dataset(target, ratio, args.seed, generated_dir)
        if ratio == 100 and row["leakage_audit"]["generated_train"] <= 0:
            raise RuntimeError(f"NEU mixed dataset has no generated train images: {target}")
        if row["leakage_audit"]["generated_val"] or row["leakage_audit"]["generated_test"]:
            raise RuntimeError(f"Generated leakage detected in NEU dataset: {row['leakage_audit']}")
        (target / "formal_manifest.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NEU formal dataset-specific DRFT-LoRA native datasets.")
    parser.add_argument("--orig-images-dir", type=Path, required=True)
    parser.add_argument("--orig-labels-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--generated-images-dir", type=Path, required=True)
    parser.add_argument("--scores-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "experiments" / "results" / "four_dataset_lora_native" / "neu_dataset_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    rows = build(parse_args())
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
