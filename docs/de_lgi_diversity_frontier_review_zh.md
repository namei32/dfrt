# DE-LGI 多样性不足的底层结构诊断与前沿文献阅读

生成时间：2026-06-11

本报告围绕当前项目内 `DE-LGI` 的生成多样性不足问题展开。文献部分共阅读 33 篇论文，覆盖扩散生成基础、局部编辑/可控生成、工业异常/缺陷合成、合成数据有效性四类方向。为避免只基于摘要判断，已下载并抽取 33 篇 arXiv PDF 正文；审计记录位于：

`D:\drft-v2\.tmp_literature_cache\full_text_audit.json`

审计记录包含每篇论文的页数、正文抽取字符数，以及 method / experiment / ablation / limitation / diversity 等关键词段落命中情况。

## 一、当前 DE-LGI 为什么多样性不明显

当前 DE-LGI 的核心公式可以概括为：

```text
z_{t-1} = z_ctx(t-1) + P(delta_raw | M_core, M_shell, M_out, E_target)
delta_raw = z_refined - propagate(z_ctx(t-1), M_out)
```

也就是说，每一步反向扩散并不是自由生成，而是先得到一个 raw update，再把这个 update 投影回“真实上下文 latent + 受能量约束的局部残差”。这给保真度带来了好处，但从底层结构上压缩了多样性。

### 1. 每一步都锚定真实上下文 latent

代码位置：

- `D:\drft-v2\generation\neu-det-pipeline\src\neu_det_pipeline\models\context_lgi.py`

关键逻辑：

```python
context_for_next = self._add_noise(pipe.scheduler, context_latents, context_noise, next_t)
delta_raw = refined - propagated_background
delta_projected, trace = _project_defect_energy(...)
latents = context_for_next + delta_projected
```

这意味着采样轨迹在每一步都会被拉回原图的 noisy latent。随机噪声虽然存在，但它只能作为局部 residual 存活；大尺度形态变化会被 `context_for_next` 抑制。

结论：当前方案是“上下文保持型局部扰动”，不是“多模态缺陷重采样”。

### 2. 能量投影是标量 RMS 约束，不是结构分布约束

当前 `_project_defect_energy` 使用：

```python
scale = target / raw_core_energy
scale = clamp(scale, min_projection_scale, max_projection_scale)
core_delta = latent_core * delta_raw * scale
```

它只控制 core 区域 high-pass residual 的整体 RMS 能量。这样会出现两个问题：

- 如果 raw update 太大，整体被缩小，细节也跟着一起缩小。
- 如果 raw update 的方向和原缺陷结构相似，RMS 变大也只是“同类纹理增强”，不会产生新的形态模式。

本次新参数生成的统计也验证了这一点：

| 指标 | 数值 |
|---|---:|
| `inside_delta_avg` | 0.014010 |
| `outside_delta_avg` | 0.008848 |
| `target_energy_avg` | 0.138284 |
| `projected_energy_avg` | 0.093832 |
| `projection_scale_avg` | 0.305334 |
| `scale < 0.5` | 48 / 50 |

虽然能量被抬高了，但 48/50 的样本仍被压到 0.5 以下，说明投影器仍然强烈限制变化。

### 3. mask 多样性来自弱形态先验，但 bbox 与标签继承锁定了自由度

当前 mask 由 `MorphologyPrior` 从真实缺陷框内提取弱形态统计，再在同一 bbox 中生成：

```python
profile = prior.sample_profile(cls_name, seed=seed)
scaled = scale_bbox(target_bbox, target_source_size, target_size)
coverage = profile.weak_coverage * rng.uniform(0.78, 1.18)
orientation = profile.orientation_deg + rng.uniform(-18.0, 18.0)
```

这给 mask 一定扰动，但仍然锁死了：

- 缺陷发生位置：沿用原 bbox。
- 缺陷尺度：受原 bbox 和 coverage clamp 限制。
- 缺陷语义：沿用原类别 prompt。
- 训练标签：`source-label-inherited`。

因此生成结果更像“在原缺陷上轻微重绘”，不是“同类缺陷的新实例”。

### 4. 默认每个样本只生成 1 个 candidate

配置中：

```python
candidates_per_sample: int = 1
```

candidate seed 虽然变化：

```python
candidate_seed = seed + sample_index * 1777 + candidate_index * 131
```

但 `candidate_index=0` 时，每个源样本只有一条轨迹。没有多候选，就没有“同源多样性选择”；质量评分也无法在质量和差异之间做 Pareto 选择。

### 5. prompt 是类别固定句式，缺少 defect attribute 变量

每类 prompt 由 `reference_caption(domain, sample.cls_name)` 得到。也就是说，`crazing`、`inclusion`、`patches` 等类别内部没有 severity、density、branching、oxidation、edge sharpness、scale roughness 等属性变化。

这会导致 cross-attention 条件几乎恒定，生成器很容易回到同一类纹理模式。

### 6. 后处理 alpha composite 再次稀释变化

当前最终合成：

```python
final = create_composite_preserving_background(original, expanded_canvas, full_mask)
```

这个设计会保护背景，但也会把边界和 shell 区域的新内容混回原图。对于 NEU 这种低对比灰度纹理，alpha 稀释会让视觉变化更小。

### 7. 评分函数鼓励“保守真实”，没有奖励“类内覆盖”

当前质量评分包含：

- background preservation
- defect strength
- boundary smoothness
- energy match
- latent background leakage

它惩罚背景扰动，也惩罚能量偏离，但没有显式奖励同类样本之间的 LPIPS / DINO / frequency / morphology diversity。因此即使生成器产生了更有差异的候选，也可能因背景或边界风险而被压低。

### 8. 当前真实图 vs 生成图对比方式本身会显得变化小

当前生成是从真实缺陷图裁剪后重绘，再与同一真实缺陷图对比。由于输入已经包含缺陷，模型需要在“已有缺陷”上做局部改写。若目标是数据增强，更合理的路径应是：

```text
正常背景图 + 缺陷分布采样 + 局部生成
```

而不是：

```text
真实缺陷图 + 同框局部重绘
```

后一种路径天然接近原图。

## 二、从底层结构看，多样性应该来自哪里

综合当前代码和文献，缺陷生成的多样性至少有五个独立自由度：

| 自由度 | 当前 DE-LGI 状态 | 结果 |
|---|---|---|
| 背景自由度 | 原图 context latent 强锚定 | 背景非常稳，但全局变化弱 |
| 位置自由度 | 继承原 bbox | 缺陷空间分布不变 |
| 形态自由度 | morphology prior 有扰动，但受 bbox/coverage 限制 | 裂纹走向/斑块轮廓变化有限 |
| 纹理自由度 | 标量 energy RMS 控制 | 多为局部对比度变化 |
| 语义自由度 | 类别 prompt 固定 | 类内风格变化不足 |

当前 DE-LGI 主要打开了“纹理能量自由度”，少量打开“形态自由度”，但没有真正打开位置、语义和多轨迹采样自由度。

## 三、文献全文阅读表

说明：以下不是摘要复述，而是结合正文的方法、实验、消融和局限段落后，对每篇论文提炼其创新点、缺点以及对 DE-LGI 的启发。链接均指向 arXiv PDF。

| # | 论文 | 创新点 | 缺点 / 局限 | 对 DE-LGI 的启发 |
|---:|---|---|---|---|
| 1 | [DDPM, Ho et al., 2020](https://arxiv.org/pdf/2006.11239) | 将去噪扩散与变分界/score matching 联系起来，证明逐步去噪可生成高质量图像。 | 采样慢；无条件 DDPM 对精确局部控制不直接友好；保真和条件控制需要额外机制。 | DE-LGI 的随机性源于扩散轨迹，但过强 context anchor 会削弱 DDPM 原本的分布覆盖能力。 |
| 2 | [DDIM, Song et al., 2020](https://arxiv.org/pdf/2010.02502) | 构造非马尔可夫反向过程，加速采样，并允许 latent 插值。 | 确定性 DDIM 会降低随机覆盖；加速和多样性需要权衡。 | 当前若使用偏确定性调度，会让同源样本更相似；可引入局部随机 DDIM eta。 |
| 3 | [Improved DDPM, Nichol & Dhariwal, 2021](https://arxiv.org/pdf/2102.09672) | 学习 reverse variance、改进噪声日程，用 precision/recall 分析覆盖。 | 指标改善不等于下游有效；仍是通用图像分布，不解决工业局部条件。 | 多样性评估不应只看视觉，应引入 coverage/recall 型指标。 |
| 4 | [Guided Diffusion, Dhariwal & Nichol, 2021](https://arxiv.org/pdf/2105.05233) | 通过 classifier guidance 提升条件生成质量，并明确 guidance 是 fidelity-diversity trade-off。 | guidance 过强会牺牲模式覆盖；需额外分类器。 | DE-LGI 的能量投影和高 CFG 类似，都是把分布推向高保真但窄覆盖区域。 |
| 5 | [Classifier-Free Guidance, Ho & Salimans, 2022](https://arxiv.org/pdf/2207.12598) | 用 conditional/unconditional score 插值替代外部分类器，实现 guidance scale 控制。 | guidance scale 大时多样性下降；双前向推理成本更高。 | 当前 `guidance_scale=7.0` 偏高，可能进一步降低同类缺陷多样性。 |
| 6 | [EDM, Karras et al., 2022](https://arxiv.org/pdf/2206.00364) | 系统拆解扩散设计空间，优化噪声参数化、预条件和采样器。 | 主要关注通用生成质量，未解决小目标工业缺陷的结构控制。 | 应把 DE-LGI 的采样器、噪声 schedule、局部重噪策略作为独立变量，而不是只调 energy scale。 |
| 7 | [Latent Diffusion Models, Rombach et al., 2021](https://arxiv.org/pdf/2112.10752) | 在 VAE latent 空间扩散，兼顾效率与高分辨率，cross-attention 支持多条件。 | VAE 压缩会损失细微纹理；小缺陷在 latent 中容易被平滑。 | NEU 缺陷很细，latent 空间中的标量能量投影容易只改低频/边缘而非真实微结构。 |
| 8 | [SDEdit, Meng et al., 2021](https://arxiv.org/pdf/2108.01073) | 通过给输入加噪再去噪，在真实性和输入保持之间做连续调节。 | 噪声太小变化弱，噪声太大结构漂移；缺少自动局部语义控制。 | DE-LGI 可以引入“缺陷区域重噪强度”，用局部 noise level 控制变化幅度。 |
| 9 | [RePaint, Lugmayr et al., 2022](https://arxiv.org/pdf/2201.09865) | 用已知区域重采样与 jump-resampling 提高 inpainting 协调性和多样性。 | 重采样增加推理时间；极小 mask 或强条件下仍可能变化有限。 | 当前 DE-LGI 缺少局部 jump-resampling；可在 core mask 内释放若干步再投影。 |
| 10 | [Palette, Saharia et al., 2021](https://arxiv.org/pdf/2111.05826) | 统一图像到图像扩散框架，并分析损失选择对多样性的影响。 | 需要成对训练/任务数据；对工业小缺陷的条件分布没有专门建模。 | 多样性可通过训练目标和条件噪声控制，不应只依赖推理后处理。 |
| 11 | [Blended Diffusion, Avrahami et al., 2021](https://arxiv.org/pdf/2111.14818) | 将 CLIP 文本引导与 DDPM 局部 mask 融合，实现自然图像局部文本编辑。 | CLIP 引导可能不稳定；ROI 内编辑容易和真实纹理不一致；速度慢。 | 局部编辑的关键不是 mask 本身，而是 mask 内要有足够开放的语义/纹理自由度。 |
| 12 | [Blended Latent Diffusion, Avrahami et al., 2022](https://arxiv.org/pdf/2206.02779) | 将 blended editing 移到 LDM latent 空间，并处理真实图重构与细 mask 问题。 | latent inversion / reconstruction error 会限制真实图编辑；细长区域仍难。 | DE-LGI 当前细裂纹变化弱，正是 latent 局部编辑在 thin mask 上的典型瓶颈。 |
| 13 | [DiffEdit, Couairon et al., 2022](https://arxiv.org/pdf/2210.11427) | 用不同文本条件的噪声预测差异自动估计编辑 mask，再做 latent inference。 | 依赖语义文本差异；细粒度工业类别描述不足时 mask 不稳。 | DE-LGI 可用条件差异估计“应该变化区域”，而非完全继承原 bbox。 |
| 14 | [Prompt-to-Prompt, Hertz et al., 2022](https://arxiv.org/pdf/2208.01626) | 通过 cross-attention map 控制文本编辑中的空间布局保持。 | 保持布局的同时会抑制形态创新；真实图编辑需 inversion。 | 当前 DE-LGI prompt 固定，attention 没有类内属性变化，导致生成模式窄。 |
| 15 | [Null-text Inversion, Mokady et al., 2022](https://arxiv.org/pdf/2211.09794) | 优化 unconditional embedding，提高真实图反演与后续编辑保真。 | 保真度提高会降低大幅编辑自由度；每图优化成本高。 | 如果 DE-LGI 继续走真实图编辑路线，过强反演/锚定会进一步压缩多样性。 |
| 16 | [Plug-and-Play Diffusion Features, Tumanyan et al., 2022](https://arxiv.org/pdf/2211.12572) | 直接注入扩散中间特征/自注意力以保持结构，实现训练自由 image-to-image。 | 结构保持太强时编辑变化不足；特征注入参数敏感。 | 当前 context latent anchor 类似结构注入，解释了“缺陷保留多、重构少”的现象。 |
| 17 | [ControlNet, Zhang et al., 2023](https://arxiv.org/pdf/2302.05543) | 冻结大模型，零卷积分支学习边缘/深度/姿态等空间控制。 | 强空间条件会让结果贴合控制图，减少自由变化；训练数据成本不低。 | 若给 DE-LGI 加 ControlNet，必须避免控制过硬，否则多样性会更低。 |
| 18 | [T2I-Adapter, Mou et al., 2023](https://arxiv.org/pdf/2302.08453) | 轻量 adapter 挖掘 T2I 模型的可控能力，可组合多条件。 | 条件组合会相互干扰；adapter 仍偏结构服从。 | 更适合作为弱条件注入，而不是硬锁定缺陷 mask。 |
| 19 | [IP-Adapter, Ye et al., 2023](https://arxiv.org/pdf/2308.06721) | 解耦文本和图像 cross-attention，实现图像 prompt 控制。 | 图像 prompt 会引入参考图偏置；若参考少，多样性受限。 | 可用多参考缺陷作为 image prompt，但要设计正/负参考采样避免复制。 |
| 20 | [CutPaste, Li et al., 2021](https://arxiv.org/pdf/2104.04015) | 简单裁剪粘贴自监督异常合成，学习异常判别表示。 | 视觉真实性有限；粘贴边界和物理缺陷机制不真实。 | 多样性来自位置、尺度、patch 来源，而当前 DE-LGI 基本没有位置自由度。 |
| 21 | [DRAEM, Zavrtanik et al., 2021](https://arxiv.org/pdf/2108.07610) | 重构网络与判别分割联合训练，用外部纹理合成异常。 | 外部纹理与真实工业缺陷存在语义差距；合成分布可能过宽。 | 缺陷多样性需要 texture source，但必须有 domain gating，不能任意贴纹理。 |
| 22 | [Natural Synthetic Anomalies, Schlüter et al., 2021](https://arxiv.org/pdf/2109.15222) | 用 Poisson blending 融合自然 patch，生成更自然的合成异常。 | patch 来源仍未必符合工业物理；复杂缺陷拓扑不足。 | 融合策略比硬合成重要；DE-LGI 的 alpha composite 应从稀释转为缺陷物理融合。 |
| 23 | [SimpleNet, Liu et al., 2023](https://arxiv.org/pdf/2303.15140) | 在特征空间加噪生成异常特征，用简单判别器学习边界。 | feature anomaly 不一定能回译成真实图像；视觉可解释性弱。 | 多样性可以在特征/能量空间定义，不必全部落实为像素形态。 |
| 24 | [AdaBLDM, Li et al., 2024](https://arxiv.org/pdf/2402.19330) | 工业缺陷生成中使用 Blended Latent Diffusion、trimap、在线 decoder adaptation。 | 模块较多；在线适配成本高；生成质量依赖 prompt/mask 设计。 | 它明确说明局部生成要分 free diffusion、editing、decoder adaptation；DE-LGI 当前缺少 free diffusion 阶段。 |
| 25 | [AnomalyPainter, Lai et al., 2025](https://arxiv.org/pdf/2503.07253) | VLLM + LDM + Tex-9K 纹理库，目标是同时提升真实性与多样性。 | 依赖外部纹理库和 VLLM；工业物理一致性仍需验证。 | 多样性不是随机噪声能解决的，需要显式 texture/attribute library 或等价的内部缺陷谱。 |
| 26 | [AnoGen, Gui et al., 2025](https://arxiv.org/pdf/2505.09263) | few-shot anomaly embedding 学习真实异常分布，再用 bbox 引导扩散生成。 | 需要少量真实异常；bbox 引导仍可能限制位置/形态；检测增益依赖后续模型。 | DE-LGI 应从真实缺陷中学习“异常嵌入分布”，而不只是能量标量。 |
| 27 | [SARD, Wang et al., 2025](https://arxiv.org/pdf/2508.03143) | 区域约束扩散冻结背景，只更新异常区域，并用 mask-aware discriminator 提升局部 fidelity。 | 仍依赖 mask；区域冻结会牺牲跨区域自然扩散和大范围变化。 | 与 DE-LGI 相似地保护背景，但 SARD 用判别 mask guidance 补偿局部纹理质量。 |
| 28 | [GLASS, Chen et al., 2024](https://arxiv.org/pdf/2407.09359) | 全局特征异常与局部图像异常联合合成，用梯度上升生成 near-distribution weak defects。 | 偏检测训练，不直接生成高保真图像；梯度方向设计依赖模型。 | 对 NEU 细微缺陷很有启发：应生成 near-boundary 难例，而不是只追求视觉大变化。 |
| 29 | [PBAS, Chen et al., 2024](https://arxiv.org/pdf/2412.17458) | 渐进边界学习、方向性 feature anomaly synthesis、边界优化。 | feature-level 合成和真实图像生成之间仍有 gap；方向边界依赖训练表征。 | DE-LGI 的 energy prior 可升级为“边界方向 prior”，让变化沿检测边界展开。 |
| 30 | [ExDD, Aqeel et al., 2025](https://arxiv.org/pdf/2507.15335) | 用 diffusion synthesis 建立正常/异常双分布记忆库，避免单一异常假设。 | 实验范围较窄；生成样本数量和质量对 memory bank 敏感。 | 当前 DE-LGI 只做图像生成，缺少“生成样本是否覆盖异常分布”的双分布反馈。 |
| 31 | [Synthetic Data from Diffusion Models Improves ImageNet Classification, Azizi et al., 2023](https://arxiv.org/pdf/2304.08466) | 系统研究扩散合成数据对 ImageNet 分类的增益，强调采样超参与筛选。 | 通用 ImageNet 与工业小缺陷差距大；合成好看不等于工业有效。 | 生成后必须做筛选和下游验证；DE-LGI 不能只看视觉对比。 |
| 32 | [TADA, Nguyen et al., 2025/2026](https://arxiv.org/pdf/2505.21574) | 只增强训练早期难学样本，避免 10-30 倍盲目扩增，并讨论 diversity 和算力问题。 | 需要训练动态信号；分类任务为主，缺陷检测需改造。 | NEU 生成不应平均给每类 50 张，而应针对检测器难例定向生成。 |
| 33 | [When Pretty Isn't Useful, Adamkiewicz et al., 2026](https://arxiv.org/pdf/2602.19946) | 发现更新更强的 T2I 模型作为训练数据生成器反而可能分布更窄，强调 aesthetics 与 data realism 不一致。 | 主要评估通用分类数据；不是工业缺陷专门研究。 | 对 DE-LGI 最关键：不能把“图像更漂亮”当作“数据更多样”，必须度量分布覆盖。 |

## 四、文献综合结论

### 结论 1：强控制通常牺牲多样性

Guided Diffusion、CFG、ControlNet、Prompt-to-Prompt、Null-text Inversion 都指向同一规律：控制越强，保真越好，但模式覆盖越窄。当前 DE-LGI 同时使用：

- 原图 context latent 锚定
- 固定 bbox / mask
- 高 `guidance_scale`
- energy projection
- background-preserving composite

因此多样性弱是结构性结果，不是单个参数没有调好。

### 结论 2：局部 inpainting 的多样性需要“释放-重锚定”循环

RePaint、SDEdit、Blended Diffusion、BLD 都说明：局部编辑如果一直贴着原图走，会很保守；如果完全释放，又会破坏背景。比较合理的是：

```text
短时释放 core latent -> 采样新形态 -> shell/background 重锚定 -> 再释放
```

当前 DE-LGI 只有“重锚定”，缺少“释放”。

### 结论 3：缺陷多样性不是单一能量，而是多维谱

工业异常文献中，CutPaste/NSA/DRAEM 打开图像空间位置和纹理来源，SimpleNet/GLASS/PBAS 打开特征空间边界方向，AnomalyPainter/AnoGen 打开语义和纹理库。它们共同说明：多样性至少应包含：

- 形态拓扑：分支数、连通分量、边界粗糙度。
- 频率谱：高频裂纹、低频斑块、方向性纹理。
- 空间关系：位置、尺度、与背景纹理方向的关系。
- 语义属性：严重程度、密度、粗糙度、氧化/暗化。
- 下游难度：是否接近检测器决策边界。

当前 DE-LGI 只显式约束了“core residual RMS energy”，维度太少。

### 结论 4：生成数据必须用 coverage/utility 筛选

Azizi、TADA、When Pretty Isn't Useful 都提醒：生成图视觉真实不代表对下游任务有用。DE-LGI 后续应加入：

- 类内 LPIPS / DINO diversity
- frequency spectrum diversity
- morphology diversity
- detector uncertainty / hard-example score
- normal-background leakage score

选择策略应从“最高质量”改为“质量-覆盖 Pareto 前沿”。

## 五、不堆模块的更好创新方向

不建议简单叠加 ControlNet、IP-Adapter、VLLM、纹理库、判别器。更好的创新应从 DE-LGI 的核心投影结构内部改造。

### 方案：Spectrum-DE-LGI，缺陷谱投影生成

核心思想：把当前单标量 `target_energy` 改为多维缺陷谱，并把投影从 RMS scale 改为谱空间约束。

当前：

```text
P(delta) = scale(delta) by core RMS
```

改为：

```text
P(delta) = project delta to class-wise defect spectrum ellipsoid
S = {energy_frequency, orientation, component_density, contrast_polarity, boundary_roughness}
```

这不是加一个新模块，而是替换 DE-LGI 的底层投影算子。

预期收益：

- 同样保护背景。
- 不再只增强原有纹理强度。
- 可生成不同频率、方向、连通结构的同类缺陷。
- 仍然保留 DE-LGI 的论文主线：defect-energy projected latent inpainting。

### 方案关键改动

1. 从真实缺陷中学习 class-wise defect spectrum，而非单个 energy scalar。

```text
profile_c = {
  q_energy_low/high,
  orientation_histogram,
  frequency_band_ratio,
  component_count_dist,
  contrast_polarity_dist,
  boundary_roughness_dist
}
```

2. 每次生成采样一个 spectrum code。

```text
s ~ p_c(spectrum | class)
```

3. 在 latent projection 中分频投影。

```text
delta_low, delta_mid, delta_high = wavelet(delta_raw)
delta_projected = P_spectrum(delta_low, delta_mid, delta_high, s)
```

4. 只在 core 内做短周期 RePaint-style release。

```text
for selected steps:
  z_core <- re-noise(z_core, local_sigma)
  z_shell/background <- context anchored
```

5. 评分函数改为质量-覆盖 Pareto。

```text
score = realism + label_consistency + background_preservation + novelty_coverage
```

其中 novelty 不和真实图逐像素差异绑定，而和同类生成集合的分布覆盖绑定。

## 六、对当前项目的直接建议

短期调参级：

1. `candidates_per_sample` 从 1 提到 4 或 8。
2. `guidance_scale` 从 7.0 降到 4.5-6.0，释放文本条件。
3. 在 core mask 内加入局部 re-noise 或 RePaint jump。
4. prompt 增加类内属性，如 severity / density / branch / roughness。
5. quality selection 加入 diversity term，不只选最高保真。

中期结构级：

1. 把 `DefectEnergyPrior` 升级为 `DefectSpectrumPrior`。
2. 把 `_project_defect_energy` 改成多频段、多方向投影。
3. 生成时使用正常背景图，而非真实缺陷图作为 source。
4. bbox 不再继承真实缺陷框，可从 class-wise spatial distribution 采样。

论文创新级：

> DE-LGI 当前创新点是“缺陷能量约束的上下文保持 latent inpainting”；下一步更强的创新应是“缺陷谱约束的多模态局部扩散”，即从标量能量控制升级为结构谱分布控制。

这比简单堆 ControlNet / IP-Adapter / 判别器更干净，也更符合当前项目的技术主线。

