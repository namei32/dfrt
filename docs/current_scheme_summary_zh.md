# 当前保留方案：DRFT-v2

Updated: 2026-06-15

当前仓库只保留 **DRFT-v2** 主线：

```text
数据准备 -> 背景投影器训练 -> DRFT-LoRA 训练 -> DRFT-v2 生成 -> real-vs-mixed 检测器评估
```

## 核心代码

核心代码位于 `generation/neu-det-pipeline/src/neu_det_pipeline`：

- `models/background_projector.py`：框感知可学习背景投影器，将缺陷框内区域投影为伪正常背景。
- `models/drft_lora.py`：训练残差场、时间步、类别条件共同调制的 DRFT-LoRA。
- `models/generator.py`：执行 DRFT-v2 生成，可选择使用背景投影器替代纯启发式背景 canvas。
- `guidance/drft.py`：构建 counterfactual canvas、缺陷残差场、context contract 与候选评分。
- `guidance/morphology.py`：根据真实样本统计生成类别形态先验 mask。
- `data/loader.py` 与 `data/resplit.py`：读取 Pascal VOC 样本，并构建 real+generated 混合数据集。
- `prompts/`：生成 DRFT-v2 所需的类别 token prompt。
- `cli.py`：暴露 `prepare`、`caption`、`train-bg-projector`、`train-drft`、`generate`、`train-yolo` 主线命令。

已删除 Reference-CMDP、Context-SGDA、DE-LGI、SRT、HDSI、HDSI-PD 等非当前方案代码。

## 两个主创新点

1. **反事实缺陷残差因子化与重组**

   DRFT-v2 不直接生成整张缺陷图，而是先估计伪正常背景，再在残差空间中建模和重组缺陷：

   ```text
   R_defect = I_src - I_bg
   I_gen = I_bg + G * Delta_defect
   ```

2. **类别-形态-原型协同的 DRFT 残差注入**

   生成过程同时使用类别残差先验、bbox 内形态场、真实残差 prototype bank，并通过 DRFT-LoRA 注入扩散 U-Net，使缺陷合成受结构、类别和真实残差模式共同约束。

## 可学习背景投影器

背景投影器是一个轻量 U-Net，输入为：

```text
RGB source + bbox mask + erase mask + context ring + box-relative x/y coordinates
```

输出为：

```text
I_bg        框内伪正常背景
U_bg        背景不确定性图
P_protect   纹理保护图
```

训练时没有真实成对正常图，因此使用当前 DRFT-v2 的 adaptive counterfactual canvas 作为软标签；推理时先生成启发式 fallback，再由学习到的 projector 输出背景，并按不确定性与 fallback 融合。

## CLI 入口

训练背景投影器：

```powershell
cd generation\neu-det-pipeline
$env:PYTHONPATH="src"
python -m neu_det_pipeline.cli train-bg-projector <dataset_root> --output-dir outputs/background_projector --train-split train --steps 500 --resolution 512
```

训练 DRFT-LoRA：

```powershell
python -m neu_det_pipeline.cli train-drft <dataset_root> --output-dir outputs/drft_lora --train-split train --steps 100
```

使用背景投影器生成：

```powershell
python -m neu_det_pipeline.cli generate <dataset_root> <guidance_dir> <drft_lora.safetensors> --mode drft-v2 --generation-split train --drft-candidates 3 --bg-projector-path outputs/background_projector/background_projector.pt
```

不传 `--bg-projector-path` 时，生成流程保持原有启发式 counterfactual canvas 行为。

## 生成记录

最近一次 NEU-DET 50 张 DRFT-v2 生成结果：

```text
D:\drft-v2\generation\neu-det-pipeline\outputs\neudet_drft_v2_50\run_drft-v2_20260615_150148
```

其中生成图像位于：

```text
D:\drft-v2\generation\neu-det-pipeline\outputs\neudet_drft_v2_50\run_drft-v2_20260615_150148\images
```
