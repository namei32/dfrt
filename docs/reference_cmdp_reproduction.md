# Reference Paper Reproduction: SCMP + CMDP

This document records the local reproduction path for:

`Context-aware defect sample generation using conditional diffusion model for surface defect inspection`
(`Measurement`, 2026, DOI: `10.1016/j.measurement.2026.121432`).

## Implemented Mapping

| Paper component | Local implementation |
|---|---|
| Bbox defect crop extraction | `ReferenceDefectCropDataset` |
| Domain captions | `A photo of a defect on the steel/fabric surface.` |
| LDM fine-tuning with frozen VAE and text encoder | `neu-det train-reference-ldm` |
| Per-class independent LDMs | `--per-class` default |
| Surrounding context masking process (SCMP) | `build_scmp_pixel_masks` |
| Expanded bbox context region | `--dilation-factor` in `generate-reference-cmdp` |
| Conditional masking diffusion process (CMDP) | `ReferenceCMDPGenerator.generate` |
| Latent fusion after each reverse step | `z_{t-1} = (1-M) * z_refined + M * b_{t-1}` |
| Mixed real+generated detector dataset | `--make-mixed-dataset` |
| Detector evaluation | existing `neu-det train-yolo` |

## NEU-DET Smoke Reproduction

Run from the repository root.

```powershell
$env:PYTHONPATH="D:\drft-v2\generation\neu-det-pipeline\src"

python -m neu_det_pipeline.cli train-reference-ldm `
  data\NEU\split\neu_det_split `
  --output-dir generation\neu-det-pipeline\outputs\reference_ldm_neu_smoke `
  --train-split train `
  --per-class `
  --steps 1 `
  --max-train-samples 1 `
  --batch-size 1 `
  --resolution 512

python -m neu_det_pipeline.cli generate-reference-cmdp `
  data\NEU\split\neu_det_split `
  generation\neu-det-pipeline\outputs\reference_ldm_neu_smoke `
  --output-dir generation\neu-det-pipeline\outputs\reference_cmdp_neu_smoke `
  --generation-split train `
  --max-samples 1 `
  --steps 4 `
  --resolution 512 `
  --make-mixed-dataset
```

The smoke commands are for validating the pipeline only. They are not paper-quality training.

## Formal Reproduction

The paper trains independent LDMs for each defect class. Use substantially more steps than the smoke run.

```powershell
python -m neu_det_pipeline.cli train-reference-ldm `
  data\NEU\split\neu_det_split `
  --output-dir generation\neu-det-pipeline\outputs\reference_ldm_neu_formal `
  --train-split train `
  --per-class `
  --steps 1000 `
  --batch-size 1 `
  --learning-rate 1e-5 `
  --resolution 512

python -m neu_det_pipeline.cli generate-reference-cmdp `
  data\NEU\split\neu_det_split `
  generation\neu-det-pipeline\outputs\reference_ldm_neu_formal `
  --output-dir generation\neu-det-pipeline\outputs\reference_cmdp_neu_formal `
  --generation-split train `
  --dilation-factor 1.3 `
  --steps 50 `
  --guidance-scale 7.0 `
  --resolution 512 `
  --make-mixed-dataset
```

Then train a detector on the generated mixed dataset:

```powershell
python -m neu_det_pipeline.cli train-yolo `
  generation\neu-det-pipeline\outputs\reference_cmdp_neu_formal\<run>\mixed_dataset\data.yaml `
  --model yolov8s.yaml `
  --epochs 300 `
  --batch 16 `
  --imgsz 640 `
  --workers 0
```

## Notes

- The local implementation follows the paper equations using a 4-channel latent diffusion model. It does not use the retained DRFT-v2 residual-field LoRA path.
- `generate-reference-cmdp` supports class-specific model directories (`model_dir/<class_name>`) and falls back to `model_dir` if a class subdirectory is missing.
- Validation and test splits remain real-only when the mixed dataset is created.
- Full five-dataset reproduction still requires dataset-specific conversion into VOC/YOLO split layouts; the code path now supports split-VOC layouts such as `images/train` and `annotations/train`.

