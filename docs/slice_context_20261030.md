# Mean Teacher + 中心残差邻切片融合

## Material Passport

- 日期：2026-10-30；来源：用户确认的方案、仓库冻结实验、公开原论文。
- 状态：模块及CPU/CUDA科学不变量已实现并检查；真实短测和效果筛查的状态以运行目录为准。
- 目标：在相同中心样本、标注预算、更新数和选模规则下，检验邻切片信息能否改善单个EMA教师。
- AI辅助文献检索、实现与核查。没有上传医学图像，未增加人工标注，不是CFF官方复现。
- 不能用软件测试、探针或多网络平均替代单网络性能证据。

## 当前架构及历史结果

历史FTP在 `model/` 中保留U-Net、APGM/PSGM、文本提示和频率分支；默认主线已经迁移到 `freqssl/`。
v3的普通U-Net有四次下采样，通道32/64/128/256/256，GroupNorm、LeakyReLU、最近邻上采样加卷积，5,682,868参数，无预训练。
Mean Teacher使用一个学生及其EMA；双学生各有独立EMA，分别测试原图交叉监督和高频增强视图。
OAR-Spine把第二学生改为组织/相对编号因子化预测，并研究顺序证据决定监督粒度。其顺序机制和后续冻结融合探针尚无通过预设筛查的证据。

| seed42、6000步，单个teacher1 | valid Dice% | 历史test Dice% |
|---|---:|---:|
| Mean Teacher | 79.785944 | 79.244 |
| 原图交叉双学生 | 79.781 | 78.817 |
| 频率交叉双学生 | 79.804 | 78.635 |

历史证据：`outputs/freqssl_v3_test_20261022/report.md`；本轮基线为
`outputs/freqssl_v3_screening_20261021/comparison/seed42/mean_teacher`。
baseline的valid union-present Dice为75.871567%。不得把不同训练预算、两教师集成或160²位置探针分数与主指标混比。

## 近一年文献及适配边界

检索窗口2025-10-29至2026-10-29，OpenAlex定向检索，并通过PubMed、arXiv及会议原文核验。
Semantic Scholar API访问未成功。检索词包括 cross slice medical image segmentation、semi-supervised medical image segmentation、vertebra segmentation、quality-guided segmentation。
这是定向筛选而非系统综述；预印本与正式发表版本分别标识，未声称查全或全球首创。

| 文献 | 核心机制 | 本项目处理 |
|---|---|---|
| [CFF，Biomedical Optics Express 2026](https://pubmed.ncbi.nlm.nih.gov/42460345/)，[所读方法全文§3](https://arxiv.org/html/2603.24115v1) | 共享编码器、相邻切片像素级权重、特征融合后预测中心 | 首轮借鉴；原文的视网膜展平、预处理和边界分支不迁入脊柱任务 |
| [G²RA-Net，2026-09预印本](https://arxiv.org/html/2609.20088v1) | 切片图关系与跨切片注意力门控 | 支持选择性上下文动机；不加入图网络或其额外损失 |
| [QG-SSL，MICCAI 2026](https://papers.miccai.org/miccai-2026/0840-Paper4232.html) | 学习图像—掩码质量，重加权伪标签或质量正则 | 保留后续候选；二类分割质量评估向20类任务的迁移未经验证 |
| [CPS4，MICCAI 2026](https://papers.miccai.org/miccai-2026/0214-Paper0946.html) | 类别提示与脊柱单元对齐，辅助Mean Teacher伪标签 | 使用MRSpineSeg和VLM预训练，分数不可直接对照SPIDER |
| [SemiGDA，CVPR 2026](https://arxiv.org/html/2604.23274v1) | 生成式图像/掩码分布对齐与双路跳接适配 | 完整框架依赖生成式先验；首轮不拆取其组件 |
| [PC-Seg，MICCAI 2026](https://papers.miccai.org/miccai-2026/0763-Paper6555.html) | 跨视图一致性、2D至3D蒸馏与共同训练 | 训练和结构改动较大，首轮仅研究局部邻片融合 |

读者推断的迁移风险：脊柱邻片同像素不一定对应相同结构，跨切片平均可能抹平边界；相邻MRI切片也不一定能提供编号所需的解剖锚点。已有误差审计表明编号错误不是主要误差，不能将所有收益或失败归于编号。

## 实现

输入 `[B,3,H,W]` 按 `s-1,s,s+1` 排列，三个切片共享原编码器；输出 `[B,20,H,W]` 仅对应中心切片。
stride4/8/16使用独立融合层；stride1/2保留中心特征。融合后的特征只送入中心解码器，不递归改变邻片编码特征。

```
w = softmax_slice(Conv1x1(concat(F_left,F_center,F_right)))
F_out = F_center + gamma * (sum_j w_j F_j - F_center)
```

adaptive每层gamma与权重卷积均零初始化，初始输出保持中心特征。mean固定w=1/3，仅学习gamma。
gamma是无约束标量，不解释为概率；可为负或超过1。自适应权重在gamma离开0后才开始得到有效梯度。
adaptive新增5,772参数，mean新增3参数。共享主干的初始化完全相同，新增参数不消费调用方的随机流。

`slice_context: none|mean|adaptive` 默认none，仅支持mean_teacher/supervised。none保留旧一通道接口；旧checkpoint缺失该字段时默认为none，可正常推理。
历史运行续训应使用其冻结源码；既有源码哈希校验不因本次改动放宽。

数据规则：

- 同病例数值切片顺序，端点复制最近切片；重复数值编号、计数不一致或缺片直接报错。
- 仅使用配置选中的影像池；截断病例的max_samples不支持上下文模式，避免偷偷扩大训练池。
- 三张切片共用中心样本的几何变换；分别进行原有归一化。仅读中心标签，未标注样本从不打开标签文件。
- 中心强增强保持旧随机种子；左右邻片使用独立、确定的种子偏移。CutMix三通道共用供体与矩形，中心伪标签同步。
- 两路邻片增加影像曝光和计算，比较控制中心更新数而非总像素曝光/GPU时间。

推理按病例创建滑动编码缓存，各真实切片编码一次，只缓存当前解码批次及其邻片。不跨病例、网络或调用复用缓存。
上下文模型在eval模式使用固定8样本的编码、融合/解码和输出微批次，尾批填零后丢弃填充输出。
GroupNorm不会混合样本，填零不会改变真实切片的统计量，但会增加少量空白计算；耗时测量包含该开销。
这避免CUDA因缓存与普通推理的批大小不同而选择不同卷积算术；同样适用于EMA教师产生伪标签。
训练学生的批处理与关闭模块的历史路径保持原实现。
`predict_spine.py`通过输入影像及可信几何产生三切片窗口，不需要任何标签，也不进行重编号或多模型集成。
邻片通道按采集数组的切片索引排列，与既有PNG数值序号一致；物理左右方向规范化若反转切片轴，先恢复该排列再推理，最后恢复输出几何。
这种排列不是对左右通道交换的不变性保证。已有来源核对证明原始/转换MHA之间仅轴置换、没有切片轴翻转；新来源应明确其采集顺序。
部署CLI沿用其关闭TF32的设置，训练/valid保持历史精度设置；两种精度的指标不能混用。

## 预设筛查及确认

配置 `configs/v3/slice_context.yaml`：seed42、6000步、batch8(4+4)、224²，warmup600、ramp1200、每500步完整valid。
损失、阈值、优化器及学习率保持原设置。mean和adaptive均从零训练。

复用历史baseline之前检查：完整协议与划分、源码快照哈希、主干初始化、真实中心样本，以及当前关闭路径与历史冻结源码三次学生/教师更新完全一致。
重新评估baseline完整valid，Dice和union-present必须重现历史结果，否则停止筛查并报告。

选择规则在运行前固定：

1. 候选Dice比baseline至少+0.3pp，且union-present不下降。
2. adaptive还须超过mean至少0.1pp；否则mean通过则选择mean。
3. 两组均不通过则停止，不延长训练或叠加新损失。
4. 通过且启用 `--confirm-on-pass` 时，固定选定模块，与新训练Mean Teacher比较seeds42/43/44、各30000步、每1000步验证、warmup1500/ramp3000。
5. 确认描述性门槛：平均Dice至少+0.3pp，至少2/3种子正向，平均union-present不下降。门槛不等于统计显著性。
6. 仅确认通过后冻结六个checkpoint哈希，进行历史test描述性评估；不利用test选模型/调参。历史test已多次查看，不能称独立新确认集。

逐病例配对bootstrap为10,000次、固定seed20261029，只反映开发验证病例差异，不是种子不确定性或独立泛化证据。
报告还包含组织Dice、20类混淆、同组织编号错误计数、伪标签覆盖、gamma、训练耗时与显存。
完整valid计时仅加载一个teacher，包含预处理、重建与指标；共享GPU单次测量不是隔离的延迟基准。

## 运行及产物

```sh
/home/zhp/.conda/envs/mamba/bin/python -m unittest discover -s tests -p test_slice_context.py -v

/home/zhp/.conda/envs/mamba/bin/python experiments/run_slice_context.py \
  --output-dir outputs/spine_slice_context_20261030_r2 --device cuda:0 --confirm-on-pass
```

脚本先冻结源码与protocol，再从快照执行。可使用 `--prepare-only` 只准备，执行 `command.json` 的argv。
目录禁止覆盖，失败不自动重试。单个训练的last.pt可按原快照、相同协议恢复；套件失败后的继续需核对已完成组与checkpoint，不能复写结果。

产物：`protocol.json`、`code/`、`results.json`、`report.md`，两组子目录、baseline和候选benchmark JSON。
通过时另有`confirmation/`和条件生成的`test_descriptive.json`。所有实际结论必须以这些完成产物为依据。

## 工程验证记录

`outputs/slice_context_checks_20261030/summary.json`记录检查及对应源码哈希。
初始CUDA真实缓存检查发现196个像素差异，`outputs/spine_slice_context_20261030`因此在694步撤销，不能用于效果选择。
固定eval微批次后，两例valid、输出批大小1/8的缓存和普通推理logits差异均为0；GPU模块测试11项通过。
另有8步真实训练与完整21例valid短测；这些结果仅证明链路可运行。

正式重启目录为`outputs/spine_slice_context_20261030_r2`，运行自己的冻结源码。
其启动后补充了仅用于无标签部署的切片索引方向修正，位于当前`freqssl/inference.py::predict_volume`；训练、验证和选模均不调用该函数，冻结运行未改动。
两例train（SPIDER_001、包含左右反转的SPIDER_006）共38张切片核验：PNG/原始MHA/转换MHA的模型输入完全相同，最终分割体素差异为0，未读标注。
可重跑`experiments/check_slice_context_cache.py`和`experiments/check_slice_context_volume.py`获得各自的影像检查记录。
