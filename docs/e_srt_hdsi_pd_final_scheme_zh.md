# E-SRT + HDSI-PD 最终方案说明

Updated: 2026-06-13

## 结论

项目中的方法线应收敛为一条清晰的研究方案：

```text
class-aware reference LDM
  -> SRT 结构残差迁移
  -> E-SRT 证据增强与背景泄漏抑制
  -> HDSI hard-but-valid 缺陷谱选择
  -> HDSI-PD 原型相位/细节注入
  -> anti-copy / visible-change / label-contract 选择器
  -> curated 50-image YOLO dataset
```

核心创新不是继续叠加更多分支，而是把结构控制、光谱难度选择、原型细节注入和质量门控合成一个可审计的生成闭环。

## 舍去或降级

以下分支保留为 baseline 或 ablation，不进入最终主方案：

| 分支 | 处理 | 原因 |
|---|---|---|
| Reference CMDP / DBT-CMDP | baseline | 复现参考论文，和当前 E-SRT/HDSI-PD 主线独立 |
| Context-SGDA | ablation | 适合诊断上下文约束，但没有学习式扩散残差主线 |
| Context-LGI | deprecated research path | 代码存在陈旧耦合，不应作为正式入口 |
| UCS / DRR / band recomposition | ablation | 分支过多，容易造成方法叙事发散 |
| SECA / bbox update | 暂不开启 | 当前目标是稳定 source-bbox label contract，避免自动框漂移 |

## 保留并优化

| 模块 | 角色 |
|---|---|
| SRT | 在源 bbox 契约内生成结构场 phi'，迁移真实缺陷残差 |
| E-SRT | 后期扩散中增强 phi' 内弱证据，抑制 phi' 外泄漏 |
| HDSI | 选择 hard-but-valid 且 source-distant 的类别缺陷谱 |
| HDSI-PD | 从同类真实缺陷原型注入相位与高频细节 |
| anti-copy selector | 拒绝源图复刻、弱可见变化和背景漂移 |
| curated export | 输出 exactly 50 images + 50 labels + data.yaml |

## 关键实现

代码入口：

```powershell
python -m neu_det_pipeline.cli generate-de-lgi `
  D:\drft-v2\data\NEU\split\neu_det_split `
  D:\drft-v2\generation\neu-det-pipeline\outputs\reference_ldm_neu_classaware_perclass_stochastic `
  --output-dir D:\drft-v2\generation\neu-det-pipeline\outputs\e_srt_hdsi_pd_neu50_final_v2 `
  --generation-split train `
  --target-generated-images 50 `
  --sample-strategy balanced `
  --preset e-srt-hdsi-pd `
  --steps 30 `
  --resolution 512 `
  --candidates 4 `
  --guidance-scale 6.0 `
  --mixed-max-generated 50 `
  --make-mixed-dataset
```

新增 preset 行为：

- 自动开启 `--ucs --srt --e-srt --hdsi --hdsi-pd --distant-spectrum`。
- 自动关闭 `--drr --band-recomposition --seca --srt-bbox-update`。
- 将目标池扩展到 `target_generated_images * 2`，再精选 50 张。
- `srt_strict_s2i_gate=false`，但 S2I 仍参与排序；真正硬门控由 E-SRT+HDSI-PD useful-change selector 执行。

## 质量门控

最终选择器以数据有效性为目标，而不是只追求视觉细节：

| 指标 | 作用 |
|---|---|
| `inside_delta` | bbox 内必须有可见干预 |
| `source_similarity` | 惩罚源图复刻 |
| `outside_delta` | 控制背景泄漏 |
| `s2i_ratio` | 鼓励变化集中在结构场 phi' |
| `hdsi_pd_selection_score` | 保留原型细节质量贡献 |

输出必须满足：

- curated 目录中 `50 images + 50 labels`。
- mixed dataset 中 `train_generated_count = 50`。
- 类别配额接近均衡。
- label class 与候选 class 一致。
- 每批生成写出 `e_srt_hdsi_pd_summary.json` 与 verification report。
