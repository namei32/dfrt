# DRFT-v2 Workspace

The workspace now keeps only the DRFT-v2 paper line:

```text
dataset-specific DRFT-LoRA training -> DRFT-v2 native generation -> formal mixed-dataset detector evaluation
```

Current Chinese scheme summary:

```text
docs/current_scheme_summary_zh.md
```

## Retained Assets

| dataset | DRFT-LoRA | DRFT-v2 native run | formal mixed dataset |
|---|---|---|---|
| GC10 | `generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_lora/gc10/drft_lora.safetensors` | `generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_v2_native/gc10/run_drft-v2_20260606_045003` | `data/GC10/formal_lora_native/gc10_drft_lora_native_ratio_100_dataset/data.yaml` |
| MT | `generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_lora/mt/drft_lora.safetensors` | `generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_v2_native/mt/run_drft-v2_20260606_083814` | `data/MT/formal_lora_native/mt_drft_lora_native_ratio_100_dataset/data.yaml` |
| NEU | `generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_lora/neu/drft_lora.safetensors` | `generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_v2_native/neu/run_drft-v2_20260608_003348` | `data/NEU/formal_lora_native/neu_drft_lora_native_ratio_100_dataset/data.yaml` |
| TILDA | `generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_lora/tilda/drft_lora.safetensors` | `generation/neu-det-pipeline/outputs/formal_dataset_specific_drft_v2_native/tilda/run_drft-v2_20260606_092918` | `data/TILDA/formal_lora_native/tilda_drft_lora_native_ratio_100_dataset/data.yaml` |

The corresponding `ratio_0_dataset` directories are retained as real-data
controls for real-vs-mixed detector comparisons.

## Runner

Run from the repository root:

```powershell
.\experiments\run_four_dataset_lora_native_formal.ps1
```

The retained three-dataset subset runner is
`experiments/run_dataset_specific_lora_yolo_formal.ps1`.

## Generation Entry

```powershell
cd generation\neu-det-pipeline
neu-det generate <dataset_root> <guidance_dir> <drft_lora.safetensors> --mode drft-v2 --generation-split train --balanced-max-samples --drft-candidates 3 --make-mixed-dataset
```

Only `drft-v2` is retained in the generation-side CLI.
