# 第二轮：提示稳健的组织教师与阶段能力门控

本文件对应已批准的第二轮协议，实验标识保留 20261104。设计 A 仍是 5,688,640 参数的邻切片 ContextUNet；设计 B 仅用于训练。实现及启动日期记录在各进程 receipt 中。结果不能在实验结束前表述为有效性证明。

## 文献适配

- [FairVLM，WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Rahman_FairVLM_Enhancing_Fairness_and_Prompt_Sensitivity_in_Vision_Language_Models_WACV_2026_paper.html)：借鉴保持组织语义的提示改写。使用固定提示库和类别平衡 JS；不复制人口统计分支，不宣称完整复现。
- [MedCLIPSeg，2026](https://arxiv.org/abs/2602.20423)：沿用现有原生 BiomedCLIP 编码器、10 层概率适配器、两级上采样与分割头。本轮不同时添加其 soft patch contrastive loss。
- [PromptSRC，ICCV 2023](https://arxiv.org/abs/2307.06948)：借鉴冻结特征约束。使用原生冻结文本特征与适配后归一化文本特征的平均余弦距离，避免误称冻结骨干即保证语义不漂移。
- [IAD，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.pdf)：仅借鉴阶段性指导思想，具体离线能力表是本任务的适配设计。

## 教师三组

T0 原句、T1 多提示与 JS、T2 在 T1 上增加冻结文本锚。每个样本两个文本视图，每个视图含四类；共享影像、几何及 MC 随机数。T0 重复原句匹配前向预算。两视图分割损失平均；T1/T2 增加 0.1 JS，T2 再增加 0.1 anchor。JS 在当前 batch 出现的每个 GT 组织内取均值，再等权平均。训练模板及留出模板以 `freqssl.prompt_revision.prompt_bank` 和冻结 protocol.json 为准。

提示选择为独立、无状态 SeedSequence(seed,step,7349)，训练 MC 使用 SHA256(revision-fit:step)+seed 的共同随机流，checkpoint 重计算保存随机状态。锚是原生文本编码器投影后的归一化 CLS；不参与梯度。与原方案相同的权重、数据、顺序、初始化及 6000 步、batch4、224²、AdamW/多项式日程。每 500 步按原句完整 valid 原生组织 Dice 选择，严格大于才替换 best。

## 病例隔离与校准

授权标签仍严格为历史前 1496 张。以 SHA256("prompt_r2:42:"+case) 排序，在完全授权的 68 例中选 14 例，共 273 张：002、008、010、013、014、026、033、041、042、044、051、054、055、066。其余 1223 张用于 pilot 教师与参考 Mean Teacher。原 2236 张未标注影像预算不变。SPIDER_069 的其余 16 张标签不可读取。

校准病例影像完全退出 pilot 训练。原生组织评估仅对 14 个完全授权病例读取体积标签；门控诊断的目标仅来自这些授权切片。参考网络每 500 步保存 step_N.pt。四套弱几何由 seed42、calibration dataset 固定顺序 index、epoch0/1/2/3 确定；不是对校准标签拟合网络。

缓存 v2：四个训练模板各两次 MC，保存 [template=4,tissue=4,H,W] float16 概率和有效区域。读取后与影像共享双线性几何并按模板重新归一化，有效区域要求插值值至少 .999。平均概率置信阈值 fg .8、bg .95，至少三个模板与均值 argmax 一致。边界为教师硬预测的 5×5 最大/最小值不同；在 CutMix 前计算，保留来源内部边界，不把混合接缝当解剖边界。

能力单元按 4 类×边界/内部×500 步阶段。只比较 EMA 20 类及聚合四类均不可靠、教师可靠且几何有效的像素；EMA 组织正确率来自四类聚合 argmax。每例合并四套几何，再按病例等权计算准确率。至少 5 例、1000 像素，教师准确率 fg≥.95/bg≥.99，优势≥.02，2000 次 seed42 病例 bootstrap 95% 区间下界>0 才开启。样本不足/未通过均关闭，不作为概率保证或统计确认。

warmup 前 600 步门控关闭。601–1000 用参考 step500，1001–1500 用 step1000，以此类推，5501–6000 用 step5500。表来源完全冻结，不根据 formal valid 或 full-label 训练过的校准标签再次调整。

## 冻结筛查与学生实验

教师须满足：校准原句平均组织 Dice 不低于 T0；任一组织下降≤0.5pp；两种留出模板各自均值下降≤1pp；至少一个前景单元开启且跨 11 个阶段累计选择前景像素/累计有效像素≥0.1%。T1/T2 都通过时，T2 valid 原句指标比 T1 高≥0.1pp 才选 T2。都不通过自动结束，写完整失败报告，不启动正式学生。

通过后，用相同结构训练 neutral pilot 并校准其门控；再从同一官方基础权重以全部 1496 张标签分别重训 text/neutral。门控从 pilot 向 full-label 模型迁移，可靠性不保证不变，报告中必须保留此限制。中性提示固定 region A/B/C/D，除词义之外训练机制相同。

S0 复用协议与更新已精确核验的邻片对照。S1 旧文本缓存+保护路由；S2 新文本缓存+保护路由；S3 S2+能力门控；S4 新中性缓存+能力门控。S1–S4 均从头 seed42、6000 步、batch8=4L+4U、224²、warmup600、ramp1200、eval500。保留 CE/Dice 和组织集合损失、0.5 组织权重、0.25 背景权重、全部像素分母，不新增 KL。

保护路由始终保留全部高置信 EMA 编号监督；组织指导仅作用于未接受 EMA 的像素。能力门控进一步检查四类置信及能力表；编号与组织像素集合互斥。EMA、图像及全部提示目标一次性共享 CutMix。

判断：S3−S0主Dice≥0.3pp、S3−S4≥0.1pp、S3−S2≥0.1pp，且相应 union-present 不下降。分别报告对照收益、文本流程优势与门控贡献。neutral 全关闭时，不把流程差异解释为文本语义证据。没有自动追加预算、多种子或 test。

## 运行与交付

主入口 `experiments/run_prompt_revision.py`。`--freeze-only` 建立不可变源码、授权校准清单、full/pilot 配置与协议；正式运行必须从输出目录的 code 执行。教师使用 medclipseg 环境，学生/校准使用 mamba 环境。所有子进程写日志、PID、退出码；阶段失败停止。暂停或中断后从最高有效 checkpoint 恢复，未完整写出的缓存/门控目录归档后按固定输入重算。启动前检查存活子进程，避免重复运行。

输出目录 `outputs/spine_prompt_revision_20261104` 包括 preflight、prior_diagnostics、teachers、calibration_caches、reference_context、gates、teacher_screen.json，以及条件通过后才产生的 caches/students。report.md/results.json 无论筛查通过或失败都会写出。通过报告含逐病例、组织 Dice、编号混淆、缺失类别误检、覆盖、显存和分阶段耗时。共享 GPU 时间不是隔离基准。

默认 prompt_source=none 和 prompt_route=legacy 保持历史行为。推理仅载入单个小网络，不读取教师、缓存或门控。新训练显式记录缓存/门控/病例清单内容哈希；旧格式缓存和检查点继续可用于推理。
