# 双学生辅助损失实验（2026-10-23）

## 审查结论与原有目标

当前双学生是两套独立 U-Net 和各自的 EMA。T1 只跟踪 S1，T2 只跟踪 S2；未标注监督方向是 **T2→S1、T1→S2**，教师始终停止梯度。S1 输入原图，S2 输入 `x + 0.5 high(x)`；频率来自几何增强后的在线有符号 Haar 残差。原实现没有特征融合、频域一致性或区域关系损失。

每个学生原有目标：

\[
L_i=\tfrac12 CE(p_i^L,y)+\tfrac12 Dice(p_i^L,y)
       +\lambda(t) CE_{mask}(p_i^U,\hat y_{T_{3-i}}).
\]

监督 Dice 使用平方和分母，并对包括背景的所有类别求平均。伪标签取对方教师 `detach→argmax`，前景阈值 0.80、背景阈值 0.95；背景 CE 权重 0.25。伪标签 CE 按**全部未标注像素数**归一，拒绝像素计零。强度增强不改变空间坐标，CutMix 同步混合图像和教师概率图。优化器只更新学生，之后各自更新 EMA。

6000 步筛查的前 601 次更新未标注权重为零，第 602 次开始非零，第 1801 次达到 1（内部从零计步）。每 500 步在全部 21 个原生 3D valid 病例选 teacher1；主指标不使用集成择优。

此前筛查没有发现双学生测试收益：单学生 test Dice 79.244%，频率双学生 78.635%。原版交叉监督确实产生了非零损失，不能再用“完全没有开启”解释失败。监督 CE 主导背景、缺少前景类别平衡和区域关系是本轮检验的假设，并非已证实原因。既有证据见 `outputs/freqssl_v3_test_20261022/report.md`。

## 文献依据与本轮改编

[Cross Teaching（MIDL 2022）](https://proceedings.mlr.press/v172/luo22b.html) 在 CNN 与 Transformer 间交换伪标签；[作者训练代码](https://github.com/HiLab-git/SSL4MIS/blob/master/code/train_cross_teaching_between_cnn_transformer_2D.py) 使用交叉伪标签 Dice。这里保留原有 EMA 和置信筛选，增加**前景宏平均伪标签 Dice**，不是复现其异构网络。

[FRCNet（MICCAI 2024）](https://papers.miccai.org/miccai-2024/paper/0245_paper.pdf) 基于单学生 Mean Teacher，使用频域一致性 FDC、多粒度区域相似性一致性 MRSC，以及像素概率 MSE。MRSC 比较区域特征的空间 Gram 矩阵。本文采用最后 decoder 特征，与论文的描述一致；[作者实际训练代码](https://github.com/NKUhealong/FRCNet/blob/main/new_model.py) 的特征提取位置存在差异。[作者 utils](https://github.com/NKUhealong/FRCNet/blob/main/utils.py) 中名为 `CriterionCosineSimilarity` 的函数实际没有启用归一化。

本轮**只移植区域关系思想**，没有加入 FDC/FEM，也不宣称复现 FRCNet。两个独立学生的隐特征通道没有固定语义对应，直接跨学生做通道 MSE 不合适；空间 Gram 对通道置换和正交基变换不敏感。仅对完整正交 DCT 系数做 MSE 又与空间 MSE 等价，不能据此声称新增了频率建模。

## 候选损失

\[
L_i=L_{sup,i}+\lambda(t)\left[
 CE_{mask,i}+0.1 Dice_{pseudo,fg,i}+1.0 L_{region,i}\right].
\]

两项均默认权重为零，旧配置保持原训练目标。本轮 YAML 显式开启权重 0.1 和 1.0；每项共享原有预热/渐增。权重在正式训练前固定，没有依据新验证/测试分数调参。

最初候选为 Dice 0.5、区域关系 0.1。在已有双学生 checkpoint 的一批训练数据上，S1 的两项加权梯度范数分别为 2.0315 和 0.00616，监督梯度为 0.6290。为了避免 Dice 压过监督项、区域项近乎无效，将权重改为 0.1 和 1.0；对应 S1 梯度约 0.4063 和 0.06159。S2 对应约 0.2183 和 0.03085，监督梯度约 0.9458。这是一次训练批次上的量级检查，不能保证整个训练期间的比例，也不是性能调优证据；初始检查记录保留。

**伪标签 Dice：** 对已经 CutMix 对齐的预测和教师目标计算。令 `w` 为与 CE 相同的置信/背景像素权重，`Y` 为硬伪标签的 one-hot，逐前景类计算：

\[
d_c=1-\frac{2\sum w p_cY_c+\epsilon}{\sum w(p_c^2+Y_c)+\epsilon},\qquad
r_c=\frac{\#accepted\ target\ pixels_c}{\max(1,\#predicted\ target\ pixels_c)}.
\]

最终对有可靠目标的前景类别求 `mean(r_c d_c)`。掩码同时进入分子和分母；可靠比例抑制“少数可信像素代表整类”的放大效应。全部不可信或只有背景时返回可反向传播的零。它有意补充 CE 的像素平均，提升小前景类别在目标函数中的地位。

**区域关系：** 将最后 decoder 特征块平均到 `g×g`，其中 `g∈{4,8,16}`。每个区域的通道向量做 L2 归一，得到 `R_g∈R^(g²×C)`；计算 `A_g=R_g R_gᵀ`，逐尺度对学生与对方教师的 `A_g` 求 MSE，再平均尺度。缩小网格、块平均和归一化均是适配现有 U-Net 的改编，论文原尺度与投影方式不同。

区域分支使用未 CutMix 的强视图额外前向，对应原始弱视图教师特征；两者空间坐标相同。像素 CE/Dice 继续使用原有 CutMix。不能直接拼接已有 Gram 矩阵作为混合图像的关系目标。该分支不增加参数或推理开销，但增加训练计算量。

日志保留 CE，并新增 `pseudo_dice_s*`、`region_s*`、各网格损失、前景可信类别数/比例、三项实际加权损失及区域教师来源。各学生分别裁剪梯度，教师不参与反向传播。

## 实验安排

根据用户要求，**不启动新的单学生对照，不重复训练已有基线**。仅启动以下三组，均从同种子初始化重新训练；不是从最佳 checkpoint 继续微调：

| 实验名 | 伪标签 Dice 权重 | 区域关系权重 |
|---|---:|---:|
| `dual_aux` | 0.1 | 1.0 |
| `dual_dice` | 0.1 | 0 |
| `dual_region` | 0 | 1.0 |

共同设置：seed42、6000 步、batch8（4 标注＋4 未标注）、1496/2236 切片、224²、AdamW 0.0003、预热600/渐增1200、每500步完整 valid、float32 严格确定性。首先启动 `dual_aux`，之后顺序运行两个拆分消融。每组超时上限4小时。

复用 `outputs/freqssl_v3_screening_20261021/comparison` 的已完成结果：

| 历史基线 | 最佳 valid teacher1 Dice | 对应步数 |
|---|---:|---:|
| 单学生 `mean_teacher` | 79.785944% | 5000 |
| 频率双学生 `dual_frequency` | 79.803749% | 4000 |

启动器检查训练配置、步数预算、数据划分和 S1 初始化是否匹配，保存引用证据的 SHA256。每完成一组，自动记录与两个历史 valid 基线的差值。源代码、配置在启动时冻结后执行，后续工作区编辑不影响排队实验。

没有 `single_aux` 组时，可以回答“加损失后双学生是否超过现有模型”，不能把差值全部归因于双学生架构。此轮为单种子筛查；训练过程中不读 test 内容，不自动按 test 调参。最终测试按用户之后的要求单独进行。

```bash
/home/zhp/.conda/envs/mamba/bin/python experiments/launch_v3.py \
  --suite auxiliary --config configs/v3/auxiliary_screening.yaml --gpu 0 \
  --reference-comparison outputs/freqssl_v3_screening_20261021/comparison \
  --output-dir outputs/freqssl_auxiliary_20261023
```

## 实施验证

验证证据位于 `outputs/freqssl_auxiliary_checks_20261023/`：

- `unit_tests.log`：61 项测试全部通过，含 CUDA、掩码梯度、空前景、通道置换、CutMix 几何、目标来源、负对照、隐藏标签隔离和精确恢复。
- `check_frozen_compatibility.json`：与上次实际执行的冻结源码对比，5 种旧方法在 CPU/CUDA 各3步，原有全部指标和所有权重逐值一致。旧频率 checkpoint 可严格加载，两教师预测逐值一致。这验证了测试场景下关闭辅助项的兼容性，不等于重新运行6000步。
- `reused_references.json`：历史基线的协议/划分/初始化核验及证据哈希。
- `probe_auxiliary_gradients.json`：用已有双学生 checkpoint 和一批训练数据检查各项梯度。只计算、不更新参数、不读取 valid/test；不是效果评估。

`real_data_smoke.json`：最终权重下，8 步真实数据训练及全部21例完整3D验证完成，耗时45.69秒，best/last checkpoint均保存。Dice在S1/S2分别4/2步非零，区域关系分别5/5步非零；置信门限拒绝全部前景时Dice为零符合设计。此短测不用于评判效果。

`startup_verification.json`：2026-10-23 15:51（Asia/Shanghai），GPU0 后台进程 **876820** 已进入 `dual_aux` 正式训练，确认至少完成1/6000步；之后顺序运行 `dual_dice`、`dual_region`。启动时全部源码快照与工作区匹配，三个正式组均为双学生。首次尝试遇到GPU占用而未创建训练进程，确认释放后正常启动。

正式运行目录：`outputs/freqssl_auxiliary_20261023/`。启动凭据为 `launch.json`，日志为 `console.log`，汇总及基线差值为 `comparison/results.json`，每组权重位于 `comparison/seed42/<arm>/`。按用户要求确认启动后停止检查，不持续监控训练；软件测试、短测完成或损失非零均不代表分割效果提升。

## Material Passport

材料：本地代码、既有训练/测试摘要、公开论文及作者实现。方法：公式核对、独立学生通道对应分析、可开关实现、合成不变量测试、冻结运行时比对、真实数据启动短测。数据处理在本地完成，未向外上传影像或标签。原始 checkpoint 与已完成实验保持原样。新模型效果尚待训练验证，不能据实现完成宣称优于单学生。

## 2026-10-26 结果补记

三组均完成6000步。按各自valid teacher1最优checkpoint，完成全部21例、481切片的原生3D test评估。Dice单项 test Dice=78.692%，区域关系单项=78.749%，两项组合=78.398%；原频率双学生=78.635%，单学生=79.244%。两个单项分别小幅提高0.057、0.115个百分点，组合降低0.236个百分点；均未超过单学生。新增损失在训练中实际非零。完整固定选模、逐病例比较、各网络结果及来源哈希见 [测试报告](../outputs/freqssl_auxiliary_test_20261026/report.md)。
