# Retained Experiment Runs

The active paper-story experiment is now limited to formal DRFT-v2 default.

## Retained Mainline

- Experiment: DRFT-v2 default
- Role: Innovation-1 baseline
- Detector: YOLOv8n
- Epochs: 200
- Seed: 0
- Dataset:
  `generation/neu-det-pipeline/outputs/drft_v2_image_level_full_c3/run_drft-v2_20260531_012045/mixed_dataset/data.yaml`
- Report:
  `runs/detect/outputs/drft_v2_image_level_full_c3/run_drft-v2_20260531_012045/yolo_runs/neu_yolov8n_default_200ep/experiment_report.json`

| mAP50 | mAP50-95 | precision | recall | F1 |
|---:|---:|---:|---:|---:|
| 0.7852 | 0.4395 | 0.7765 | 0.7089 | 0.7411 |

## Retained Supporting Evidence

- GC10 generalization records:
  `experiments/results/gc10_drft_v2_generalization_runs.csv`
  and `experiments/results/gc10_drft_v2_generalization_summary.csv`
