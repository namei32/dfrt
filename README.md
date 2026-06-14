# DRFT-v2 Workspace

This workspace is scoped to the retained DRFT-v2 data-augmentation line.

## Active Scheme

The retained method is dataset-specific DRFT-LoRA training followed by
DRFT-v2 native generation.

Chinese scheme summary:

```text
docs/current_scheme_summary_zh.md
```

## Retained Generation Artifacts

Dataset-specific DRFT-LoRA weights:

```text
generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_lora/gc10/drft_lora.safetensors
generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_lora/mt/drft_lora.safetensors
generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_lora/neu/drft_lora.safetensors
generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_lora/tilda/drft_lora.safetensors
```

DRFT-v2 native generation records:

```text
generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_v2_native/gc10/run_drft-v2_20260606_045003
generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_v2_native/mt/run_drft-v2_20260606_083814
generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_v2_native/neu/run_drft-v2_20260608_003348
generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_v2_native/tilda/run_drft-v2_20260606_092918
```

Formal mixed datasets:

```text
data/GC10/formal_lora_native/gc10_drft_lora_native_ratio_100_dataset/data.yaml
data/MT/formal_lora_native/mt_drft_lora_native_ratio_100_dataset/data.yaml
data/NEU/formal_lora_native/neu_drft_lora_native_ratio_100_dataset/data.yaml
data/TILDA/formal_lora_native/tilda_drft_lora_native_ratio_100_dataset/data.yaml
```

The matching `ratio_0_dataset` directories are retained as real-data controls.

## Main Runner

```powershell
.\experiments\run_four_dataset_lora_native_formal.ps1
```

The three-dataset DRFT-v2 subset runner is retained at
`experiments/run_dataset_specific_lora_yolo_formal.ps1`.

## Generation Entry

```powershell
cd generation\neu-det-pipeline
neu-det generate <dataset_root> <guidance_dir> <drft_lora.safetensors> --mode drft-v2 --generation-split train --balanced-max-samples --drft-candidates 3 --make-mixed-dataset
```

Only `drft-v2` is retained as a generation mode.
