# 当前保留方案：DRFT-v2

Updated: 2026-06-15

当前工作区只保留 **DRFT-v2** 主线：

```text
数据集专属 DRFT-LoRA 训练 -> DRFT-v2 native 生成 -> real-vs-mixed 检测器评估
```

## 保留代码

核心源码位于 `generation/neu-det-pipeline/src/neu_det_pipeline`：

- `models/drft_lora.py`：训练残差场/时间步/类别条件的 DRFT-LoRA。
- `models/generator.py`：使用 DRFT-v2 原生 inpainting 生成缺陷样本。
- `guidance/drft.py`：构建 counterfactual canvas、缺陷残差场、context contract 与候选评分。
- `guidance/morphology.py`：根据真实样本统计生成类别形态先验 mask。
- `data/loader.py` 与 `data/resplit.py`：读取 Pascal VOC 样本，并构建 real+generated 混合数据集。
- `prompts/`：生成 DRFT-v2 所需的类别 token prompt。
- `cli.py`：只暴露 `prepare`、`caption`、`train-drft`、`generate`、`train-yolo` 五个主线命令。

已删除 Reference-CMDP、Context-SGDA、DE-LGI、SRT、HDSI、HDSI-PD 等非当前方案代码。

## 保留资产

数据集专属 DRFT-LoRA 权重：

```text
D:\drft-v2\generation\neu-det-pipeline\outputs\formal_dataset_specific_drft_lora\gc10\drft_lora.safetensors
D:\drft-v2\generation\neu-det-pipeline\outputs\formal_dataset_specific_drft_lora\mt\drft_lora.safetensors
D:\drft-v2\generation\neu-det-pipeline\outputs\formal_dataset_specific_drft_lora\neu\drft_lora.safetensors
D:\drft-v2\generation\neu-det-pipeline\outputs\formal_dataset_specific_drft_lora\tilda\drft_lora.safetensors
```

DRFT-v2 native 生成记录：

```text
D:\drft-v2\generation\neu-det-pipeline\outputs\formal_dataset_specific_drft_v2_native\gc10\run_drft-v2_20260606_045003
D:\drft-v2\generation\neu-det-pipeline\outputs\formal_dataset_specific_drft_v2_native\mt\run_drft-v2_20260606_083814
D:\drft-v2\generation\neu-det-pipeline\outputs\formal_dataset_specific_drft_v2_native\neu\run_drft-v2_20260608_003348
D:\drft-v2\generation\neu-det-pipeline\outputs\formal_dataset_specific_drft_v2_native\tilda\run_drft-v2_20260606_092918
```

生成数量：

| Dataset | Generated images |
|---|---:|
| GC10 | 1596 |
| MT | 271 |
| NEU | 1258 |
| TILDA | 280 |

正式评估数据集入口：

```text
D:\drft-v2\data\GC10\formal_lora_native\gc10_drft_lora_native_ratio_100_dataset\data.yaml
D:\drft-v2\data\MT\formal_lora_native\mt_drft_lora_native_ratio_100_dataset\data.yaml
D:\drft-v2\data\NEU\formal_lora_native\neu_drft_lora_native_ratio_100_dataset\data.yaml
D:\drft-v2\data\TILDA\formal_lora_native\tilda_drft_lora_native_ratio_100_dataset\data.yaml
```

同目录下的 `ratio_0_dataset` 是真实数据对照，用于 real-vs-mixed 检测器对比。

## 入口约束

生成侧 CLI 只保留 `drft-v2` 模式：

```powershell
cd generation\neu-det-pipeline
neu-det generate <dataset_root> <guidance_dir> <drft_lora.safetensors> --mode drft-v2 --generation-split train --balanced-max-samples --drft-candidates 3 --make-mixed-dataset
```

正式 runner：

```powershell
.\experiments\run_four_dataset_lora_native_formal.ps1
```
