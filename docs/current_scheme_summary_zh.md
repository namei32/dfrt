# 当前保留方案

Updated: 2026-06-10

当前工作区只保留 **DRFT-v2** 主线：先训练数据集专属 DRFT-LoRA，再使用 DRFT-v2 native 生成缺陷样本，最后构建 real-vs-mixed 数据集做正式检测器评估。

## 保留内容

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

同目录下的 `ratio_0_dataset` 是对应真实 split 对照，用于 real-vs-mixed 检测对比。

## 入口约束

`generation/neu-det-pipeline` 的生成入口只保留 `drft-v2` 模式。正式 runner 使用：

```powershell
.\experiments\run_four_dataset_lora_native_formal.ps1
```
