from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _materialize_data_yaml(data_yaml: Path, output_root: Path, name: str) -> Path:
    text = data_yaml.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False
    out_lines: list[str] = []
    for line in lines:
        stripped = line.strip().lstrip("\ufeff")
        if stripped.startswith("path:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'")
            if value in {"", "."}:
                out_lines.append(f"path: {data_yaml.parent.resolve().as_posix()}")
                changed = True
            else:
                out_lines.append(line)
            continue
        out_lines.append(line)
    if not changed:
        return data_yaml
    out_dir = output_root / "data_yamls"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.yaml"
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return out_path


def _model_arg(model: str) -> str:
    model_path = Path(model)
    return str(model_path.resolve()) if model_path.exists() else model


def _run_experiment(
    *,
    data_yaml: Path,
    name: str,
    output_root: Path,
    model: str,
    epochs: int,
    patience: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    seed: int,
    skip_existing: bool,
) -> Path:
    package_root = _package_root()
    save_dir = output_root / name
    report_path = save_dir / "experiment_report.json"
    if skip_existing and report_path.exists():
        return report_path

    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{name}.out.log"
    stderr_path = logs_dir / f"{name}.err.log"

    env = os.environ.copy()
    src_path = str(package_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "neu_det_pipeline.cli",
        "train-yolo",
        str(data_yaml.resolve()),
        "--model",
        _model_arg(model),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch",
        str(batch),
        "--imgsz",
        str(imgsz),
        "--project",
        str(output_root.resolve()),
        "--name",
        name,
        "--device",
        device,
        "--workers",
        str(workers),
        "--seed",
        str(seed),
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(
            cmd,
            cwd=package_root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    if not report_path.exists():
        raise FileNotFoundError(f"Expected report not found: {report_path}")
    return report_path


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], key: str) -> float | None:
    value = report.get("test_metrics", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _write_summary(report_paths: dict[str, Path], output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    for name, path in report_paths.items():
        report = _load_report(path)
        reports[name] = report
        dataset, variant = name.rsplit("_", 1)
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "mAP50": _metric(report, "mAP50"),
                "mAP50-95": _metric(report, "mAP50-95"),
                "precision": _metric(report, "precision"),
                "recall": _metric(report, "recall"),
                "f1": _metric(report, "f1"),
                "report_path": str(path.resolve()),
            }
        )

    deltas: list[dict[str, Any]] = []
    by_dataset = {row["dataset"] for row in rows}
    for dataset in sorted(by_dataset):
        original = next((row for row in rows if row["dataset"] == dataset and row["variant"] == "original"), None)
        mixed = next((row for row in rows if row["dataset"] == dataset and row["variant"] == "mixed"), None)
        if original is None or mixed is None:
            continue
        delta_row: dict[str, Any] = {"dataset": dataset}
        for key in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
            ov = original.get(key)
            mv = mixed.get(key)
            delta_row[f"delta_{key}"] = (mv - ov) if isinstance(ov, float) and isinstance(mv, float) else None
        deltas.append(delta_row)

    csv_path = output_root / "validation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "variant", "mAP50", "mAP50-95", "precision", "recall", "f1", "report_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    delta_path = output_root / "validation_deltas.csv"
    with delta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "delta_mAP50", "delta_mAP50-95", "delta_precision", "delta_recall", "delta_f1"],
        )
        writer.writeheader()
        writer.writerows(deltas)

    json_path = output_root / "validation_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "deltas": deltas,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(json_path.resolve()), "csv": str(csv_path.resolve())}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8n original-vs-mixed validation.")
    parser.add_argument("--neu-original", type=Path, required=True)
    parser.add_argument("--neu-mixed", type=Path, required=True)
    parser.add_argument("--gc10-original", type=Path, required=True)
    parser.add_argument("--gc10-mixed", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=str(_package_root() / "yolov8n.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    experiments = {
        "neu_original": args.neu_original,
        "neu_mixed": args.neu_mixed,
        "gc10_original": args.gc10_original,
        "gc10_mixed": args.gc10_mixed,
    }
    report_paths: dict[str, Path] = {}
    for name, data_yaml in experiments.items():
        resolved_yaml = _materialize_data_yaml(data_yaml, args.output_root, name)
        report_paths[name] = _run_experiment(
            data_yaml=resolved_yaml,
            name=name,
            output_root=args.output_root,
            model=args.model,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            skip_existing=args.skip_existing,
        )
    _write_summary(report_paths, args.output_root)


if __name__ == "__main__":
    main()
