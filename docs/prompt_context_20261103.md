# 邻切片融合与文本组织提示教师

研究任务：SPIDER 20 类半监督分割。设计 A 提供邻切片影像上下文；设计 B 提供训练阶段的外部组织语义。最终部署仍是一个 5,688,640 参数的 ContextUNet，输入相邻三片，输出中心片 20 类。

已有邻片模块三种子主 Dice 平均提高 0.4313pp，但 union-present 下降 0.2156pp，未通过此前确认门槛。本项目不将其改写为已全面超过 baseline 的方法。

## 文献来源与适配

| 来源 | 借鉴 | 本项目适配 |
|---|---|---|
| [MedCLIPSeg](https://arxiv.org/abs/2602.20423)；[代码](https://github.com/HealthX-Lab/MedCLIPSeg) | 冻结 BiomedCLIP 编码器，PVL 双向随机适配，像素与文本查询产生分割 | 四个提示共同产生四类 softmax；保留原生编码器；屏蔽文本 padding；训练损失固定四类 CE/Dice |
| [CPS4](https://arxiv.org/abs/2606.15802) | 适配视觉语言模型，为 Mean Teacher 提供额外监督 | 提供组织集合目标，不将文本强行解释成 SPIDER 相对编号 |
| [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | 医学图文预训练 | 使用官方基础权重和归一化，原始灰度片复制为 RGB；不使用旧 SPINE 分割权重 |

这些属于有文献依据的研究适配，不宣称共享编码、跨模态融合或集合监督原理本身首次提出。预训练数据来源是 PMC-15M；未进行逐图重叠排查，不将其表述为已证实零重叠。

## 固定提示与组织映射

1. 标签 0：`Background and other tissues in a sagittal spine MRI.`
2. 标签 1–9：`Vertebrae in a sagittal spine MRI.`
3. 标签 10：`The spinal canal in a sagittal spine MRI.`
4. 标签 11–19：`Intervertebral discs in a sagittal spine MRI.`

中性对照依次使用 `Region A/B/C/D in a sagittal spine MRI.`，固定映射，从相同基础权重及适配器初始化独立训练。对照检验解剖提示语义的额外作用；不能将其称为完全没有文本的模型。

## 教师适配与缓存

固定基础版本 `9f341de24bfb00180f1b847274256e9b65a3a32e`，ViT-B/16 和 PubMedBERT 冻结且保持 eval。10 个融合层（零基索引 1–10）、适配维 256、两级上采样、mask head。仅这些适配参数训练；采用激活重计算节省显存，随机状态在重计算中保留。

两种教师均只读取授权的前 1496 张训练标签。6000 步，batch 4，224²；AdamW lr=0.0003、weight_decay=0.0001；多项式衰减指数 0.9，梯度裁剪 12，float32。损失为 0.5 CE + 0.5 squared-denominator soft Dice（含背景）。图像及标签共享原有几何增强；不添加旧实现的图文对比损失。

每 500 步完整验证 21 例。每个组织先在 GT 存在的病例上平均原生 3D Dice，再平均三种组织；同分保留更早检查点。验证与缓存均使用 8 次随机预测，固定批次 4 并补零尾批；MC 种子由批次有序文件名和 seed42 确定。采样恢复调用前 RNG，不改变训练随机流。

冻结最佳教师，为 2236 张未标注切片缓存四类概率、硬组织目标、可信及有效区域、投票数。平均预测前景≥0.8／背景≥0.95，且至少 7/8 次与平均预测组织一致才可信；该规则不是已校准的正确概率。缓存的概率保存 float16；组织目标及门控在 float32 预测上确定后保存，不由量化概率重新决定。

缓存清单记录基础权重、适配检查点、提示、组织映射、源图像、授权标签及配置来源。训练加载检查数据划分、文件散列、缓存散列和协议；输入或缓存变化时报错。无标签缓存入口不读取掩码。验证仅用于选模；test 仅检查划分文件名隔离，不读取其图像或标签。

## 学生监督

文本不可信时沿用 EMA。文本可靠且 EMA 达标、组织一致时采用原 20 类目标；文本可靠而 EMA 不达标或组织冲突时使用组织集合目标：

`L_tissue = -log(sum(p_student[c] for c in tissue_class_set))`。

两类监督掩码互斥，按全部像素平均，背景权重均为 0.25。总损失为 `L_sup + ramp * (L_EMA_filtered + 0.5 L_tissue)`。组织目标不规定集合内各相对编号的分配；它不能直接纠正同组织内错误编号。

组织目标和掩码使用与图像相同的仿射矩阵及翻转，按最近邻插值；有效掩码使填充区域不参与提示监督。图像、EMA 目标与提示侧信息打包经过同一次 CutMix，保证供体和矩形相同。逐切片归一化、中心增强随机流、学生和 EMA 结构保持原路径。

配置接口：`prompt_source: none | text | neutral`、`prompt_cache`、`prompt_weight: 0.5`；默认关闭。缓存仅在训练数据构造时加载，部署加载检查点不检查缓存存在性，也不导入视觉语言模型依赖。

## 三组配对筛查

三组均从头训练：邻片对照、邻片＋文本教师、邻片＋中性提示教师。seed42，6000 步，batch8=4L+4U，224²，warmup600、ramp1200，每 500 步完整验证 21 例。学生由 valid 原生 20 类 GT-present Dice 选模，保留 union-present 指标作为护栏。

门槛：文本组比邻片对照主 Dice 至少 +0.3pp，union-present 不下降；同时比中性对照至少 +0.1pp，union-present 不下降，才通过文本语义筛查。若两种教师均改善而文本优势不足，只报告额外预训练教师可能有效。单种子通过不等于确认；完整消融、多种子和 test 属于后续阶段。

输出包含逐病例差值及病例 bootstrap 区间、组织 Dice、编号混淆、缺失类别误预测体素、监督覆盖、每阶段耗时及峰值显存。教师准备、缓存生成、学生训练和完整体积推理分别计时；共享 GPU 的单次计时不作为隔离性能基准。

## 运行与恢复

教师使用 `/home/zhp/.conda/envs/medclipseg/bin/python`，学生使用 `/home/zhp/.conda/envs/mamba/bin/python`。先运行 `experiments/check_prompt_teacher.py` 和单元测试，再用 `experiments/run_prompt_context.py --freeze-only` 冻结源代码及协议，正式管线从输出目录的 `code` 运行。

每个阶段分别保留日志、PID、退出码和状态。教师每 100 步保存可恢复检查点，学生沿用原验证检查点；均保存协议和来源校验。恢复重放未写入检查点的尾部更新，保留旧日志证据。中断的缓存保留于 recovery 后按固定种子重新生成。管线完成后自动生成 `report.md` 和 `results.json`，不会自动增加训练预算或访问 test。

## 实现验证

自动检查覆盖：低可信提示回退的 loss/梯度等价、集合监督的精确目标、监督掩码互斥、MC 投票阈值、CutMix 同步、仿射及填充同步、缓存来源变化拒绝、未标注标签不可读、学生/EMA 更新、断点恢复和无缓存推理。真实 BiomedCLIP 检查覆盖原生编码器等价、冻结参数不变、适配器梯度、两种提示初始化相同及 MC 可复现。

算法代码与测试通过只能证明实现约束，模型收益以三组完整实验记录为准。
