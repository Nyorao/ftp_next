# FTP Mean Teacher 实验版

代码位于 `/home/zhp/ftp_next`，视觉 U-Net 从零训练。当前阶段先关闭 APGM/PSGM，验证频率双学生；原 FTP 提示模块及 BERT 资产仍可用于后续消融。

这是一个可训练的迁移原型。已完成单元测试和真实 SPIDER 数据短流程验证，尚未完成训练或证明性能提升。

## 当前：无提示模块的频率双学生（2026-10-18）

**2026-10-19 更新**：按用户要求，两个双学生实验已从头重启为 batch=34（17+17），200 轮共 17600 步；单学生仍为 batch=4、149600 步。两双学生可在相同批量下比较；与单学生的比较包含 batch/优化更新次数差异。最新目录和预算见 `configs/frequency_three/plan.json` 与三组准备记录，以下原始统一 batch 预算为历史设置。

配置为 `configs/frequency_dual.yaml`：APGM/PSGM 均关闭，不加载 BERT，不计算提示监督。S1 输入 RAW+LF，S2 输入 RAW+HF；教师仅由 S1 EMA 更新。两个学生关闭提示时各解码一次，避免重复更新 BN。

按用户最新要求，`experiments/compare_frequency.py` 默认运行三组：`anchor_raw` 原图单学生、`dual_raw` 两学生均用原图且有反馈、`dual_frequency` LF/HF 双学生且有反馈。独立配置位于 `configs/frequency_three/`。统一使用完整既有 1496/2236 张训练池，**200 个 epoch（当前 batch=4、标注 batch=2 时共 149600 步）**，seed=42；每 5 轮做完整 valid 3D 验证。保持 S2→S1 单向反馈：至少预热 50 轮，两个学生验证 Dice 连续两次均达到 70% 后才启用，再渐增 20 轮；未达标保持关闭。测试集不参与训练、反馈门控或选模。

2026-10-19 用户授权将剩余两组并行启动：原图单学生继续使用 GPU 3，原图双学生使用 GPU 0，LF/HF 双学生使用 GPU 1。各组目录和进程见 `outputs/frequency_three_seed42_run1/parallel_dispatch.json`；请以各自目录的结果与日志判断状态。原串行入口已设置路径防重复保护，其最终调度状态不代表三个独立实验的状态。当前三组及启动约定见 [三组准备记录](docs/frequency_three_prepared_20261018.md)，文献依据见 [频率双学生方案](docs/frequency_dual_plan_20261018.md)。

旧双学生 pilot、probe、对应体数据评估与结果报告已按用户要求删除，清单见 `docs/training_cleanup_20261018.json`。三例 SPIDER spacing 修正证据保留在 `data/spider_geometry_trace.json` 与 `data/spider_geometry_audit.json`。2026-10-08 的独立基础 smoke 记录不属于此次删除范围。

## 历史实现：含提示模块的单教师、独立双学生（2026-10-15）

双学生设计、待解决问题和预先固定的实验方案见 [docs/dual_student_design.md](docs/dual_student_design.md)。Mean Teacher 是 EMA 教师训练机制；本项目的视觉 backbone 仍是从零训练的 U-Net。

```text
RAW + LF → S1: U-Net + APGM ──EMA──→ 唯一教师 T
                   ↑                    │
                   │ 筛选后的软预测      ├─一致性目标→ S1
RAW + HF → S2: 独立 U-Net + PSGM         └─一致性目标→ S2
```

S1/S2 的编码器、解码器、BN 和提示模块各自独立。教师只跟随 S1；S2 通过未标注像素上的 `KL(stopgrad(S2) || S1)` 影响 S1。反馈经过 warm-up，且只选择 S2 预测为前景、置信度达到阈值并高于 S1 的像素。APGM/PSGM 内部运算、预训练冻结 BERT 和原类别向量保持不变。

`architecture` 有三种设置：

| 设置 | 可训练网络 | 教师 | 用途 |
|---|---|---|---|
| `shared` | LF/HF 共享一个 U-Net | 整个共享学生的 EMA | 保留此前迁移版 |
| `anchor` | S1 LF/APGM | S1 的 EMA | 双学生的配对单学生基线，要求 `feedback_weight=0` |
| `dual` | 独立 S1 LF/APGM、S2 HF/PSGM | S1 的 EMA | `feedback_weight=0` 为无反馈对照，正值启用反馈 |

双学生完整训练配置为 `configs/ftp_dual.yaml`；使用 `train.py --config` 指定。默认 batch=12 未做显存验证，可先用 `--batch-size 2 --labeled-bs 1 --num-workers 0`。反馈默认第 5 epoch 开始、20 epochs ramp-up、权重 0.25、置信度阈值 0.7、margin 0.05；这些是起始设置，尚未证实最优。

三组真实数据短程对照入口（输出目录必须是新目录）：

```bash
cd /home/zhp/ftp_next
/home/zhp/.conda/envs/mamba/bin/python experiments/compare_dual.py \
  --device cuda:1 \
  --output-dir /home/zhp/ftp_next/outputs/dual_pilot_new
```

默认顺序运行 `anchor`、`dual_no_feedback`、`dual_feedback`，各 240 步，种子 42；使用相同的 48/72 张有标注/未标注切片、最多 16 个独立验证患者，保留完整模型和 224 输入。该入口专用于短程预实验，压缩了一致性和反馈的启动日程，不能直接当作完整训练配置。`--steps`、`--seed`、`--arms` 可用于新的预实验，实际参数会写入 manifest。默认总训练时限 30 分钟，超时结果标为 incomplete。

产物包括样本清单、源码快照及 hash、各组逐步损失/反馈覆盖率、固定最后一步 checkpoint、教师/S1/S2/集成的独立指标，以及无反馈对照的 S1/教师参数差异。只有单教师指标改善才能说明双学生反馈帮助了最终教师；双学生集成需单独报告。

后台启动可用 `experiments/launch_compare.py --gpu 1 --output-dir /home/zhp/ftp_next/outputs/dual_pilot_background_new`。它先检查空闲显存，再启动一个顺序运行各组的进程，保存 `launch.json`（PID/命令/GPU）和 `console.log`；可用 `--arms anchor dual_no_feedback` 只启动部分组。不会自动分析或发布实验结论。

为确保 GPU 配对实验可比，`deterministic: true` 现启用严格确定性。当前 PyTorch 2.0.1 的 CUDA 双线性上采样反传不支持该模式，因此仅训练时将该算子放到 CPU，保留 bilinear/align_corners=False；CE 采用等价的 log_softmax/gather/mean。其余网络仍在 GPU，APGM/PSGM 不变。此路径增加数据传输成本。无反馈组若不能逐元素复现 anchor，预实验入口会停止后续组。

截至 2026-10-15，17 项测试通过，包含 GPU 严格确定性下的三步配对负对照。该阶段训练输出已在 2026-10-18 清理，不作为本轮频率消融结果。

双学生 checkpoint 的评估使用 `--network teacher`、`student1`、`student2` 或 `students_mean`，分别对应 LF、LF、HF 和概率平均。教师只支持 LF 路径。下面的 `--branch hf/mean` 说明仅适用于 `shared` 架构。

## 此前共享学生结构（architecture=shared）

```text
                          学生 FTPStudent
 [原图, LF] ─┐             ┌─ LF 瓶颈 → APGM → 共享解码器 → lf_prompt
             ├─ 共享 U-Net 编码器
 [原图, HF] ─┘             └─ HF 瓶颈 → PSGM → 共享解码器 → hf_prompt
                               原瓶颈 → 共享解码器 → lf_plain / hf_plain

                          教师 FTPStudent
            与学生结构完全相同（含 APGM 和 PSGM），初始权重复制自学生
            参数仅由 EMA 更新；eval 模式；无梯度；输出未标注样本的目标
```

- **视觉网络**：随机初始化的 U-Net，五次降采样；224×224 输入的瓶颈是 768×7×7。LF/HF 是共享编码器和解码器的两个视图，不是独立优化的两个异构网络。学生和教师各有一套完整 U-Net 和提示模块。
- **APGM**：复制原 `model/fusion.py`，只删除未使用的导入，类定义和计算逻辑不变。
- **PSGM**：提取原 `ConvNeXt.psgm`，保留 768→192→48→20 的融合结构。类别文本向量注册为 buffer，随设备移动和 checkpoint 保存。默认配置下的输出及输入梯度已与原方法对照。
- **文本**：保留原 BiomedVLP-CXR-BERT-general 和原 `.pth` 类别向量。固定提示词与原数据集 `mode='fixed'` 一致；BERT 冻结并以 eval 模式计算一次最后层特征。模型保存这些特征，推理不需要重新加载 BERT。固定特征不包含逐样本标签信息。
- **预训练依赖**：只读取本地文本模型及类别向量，不加载 Swin-UMamba、ConvNeXt 或其他视觉 checkpoint，不发起模型下载。

原 FTP 的两个视觉编码器、CPS 训练和视觉预训练同时被替换；共享 U-Net 也是本次结构变化。因此，这个版本不能单独用于证明“去掉预训练”的因果收益。后续需要分别控制这些因素做消融。

## 共享的基础训练目标

有标注样本仍用 `0.4 × CE + 0.6 × Dice`，分别监督 APGM/PSGM 输出。Dice 使用原 FTP 的平方分母形式。

原双向 CPS 被教师到学生的软一致性替换：在未标注样本上分别比较 LF/HF 提示预测的 softmax，损失为每像素类别概率差的平方和，再对像素求均值。权重通过 sigmoid ramp-up 逐步增加。

原提示监督保留：学生的 `lf_prompt` / `hf_prompt` 产生 stop-gradient 硬目标，监督对应的 `lf_plain` / `hf_plain`。有监督权重 `max(1, 1.25 - 0.005 × epoch)`、提示监督权重 `min(0.25, 0.005 × epoch)` 沿用原代码。因此 epoch=0 时提示监督权重为零。

每一步仅优化学生，然后更新教师参数：`teacher ← α × teacher + (1−α) × student`。默认 α 上限 0.99，初期使用 `min(α, 1−1/(updates+1))`；第一次更新完整复制学生参数。BatchNorm 运行统计、整数计数和固定文本 buffer 直接从学生复制，教师前向不会自行更新 BatchNorm。

数据集几何增强由学生和教师共享；之后分别添加强弱不同的强度噪声，保持像素对应关系。不会对两个目标坐标系独立做几何变换。

## 运行

现有 `/home/zhp/.conda/envs/mamba/bin/python` 已具备所需依赖，本次验证使用 PyTorch 2.0.1。其他环境可按 `requirements.txt` 安装依赖。

配置在 `configs/ftp_mt.yaml`。默认沿用原 SPIDER 路径、20 类、112 个文本 token、224 输入、前 1496 个排序训练样本有标注、总计 3732 个训练样本，以及 batch_size=12 / labeled_bs=6。保存实际选用的文件列表供复现。

完整训练（在 CUDA 可用的环境中）：

```bash
cd /home/zhp/ftp_next
CUDA_VISIBLE_DEVICES=0 /home/zhp/.conda/envs/mamba/bin/python train.py \
  --config /home/zhp/ftp_next/configs/ftp_mt.yaml \
  --device cuda:0 \
  --output-dir /home/zhp/ftp_next/outputs/ftp_mt_seed42
```

两步真实数据检查（CPU 可运行，batch_size=2，完整 224 输入和默认模型，不保存大 checkpoint）：

```bash
cd /home/zhp/ftp_next
/home/zhp/.conda/envs/mamba/bin/python train.py \
  --smoke-test --device cpu \
  --output-dir /home/zhp/ftp_next/outputs/smoke_new
```

输出目录若已有训练结果会拒绝覆盖，请为不同实验指定不同目录。训练入口支持 `--max-iterations`、`--max-epochs`、`--num-workers`、`--batch-size`、`--labeled-bs`、`--image-size`、`--base-channels`。首次使用建议先运行 smoke test。

训练产物：

- `best.pt`：验证集最佳教师分数对应的完整学生/教师状态及优化器。
- `last.pt`：最新 epoch（或达到迭代上限时）的完整状态。
- `config.json`、`labeled_files.txt`、`unlabeled_files.txt`：实际配置与训练划分。
- `train.log`、`metrics.jsonl`、`summary.json`：损失、EMA 更新和验证记录。

训练期间只访问 train/valid；test 必须单独评估。未标注训练样本不会打开 label 文件，损失也只索引有标注样本的真实标签。

逐切片快速评估（仅作运行诊断，不能作为 SPIDER 的正式整体结果）：

```bash
/home/zhp/.conda/envs/mamba/bin/python /home/zhp/ftp_next/evaluate.py \
  --checkpoint /home/zhp/ftp_next/outputs/ftp_mt_seed42/best.pt \
  --split test --network teacher --branch lf --device cuda:0
```

默认评估 LF+APGM 路径，与原 FTP 采用的推理路径对应。可用 `--network student` 评估学生，或用 `--branch hf/mean` 评估 HF/两路概率平均；这些是不同设置，应分别报告。可通过 `--root-path` 更换数据位置。

Dice 沿用原 FTP 的统计约定：忽略 GT 中不存在的前景类别，先对每张切片求均值，再对切片求均值。HD95 在原标签分辨率上按像素距离计算，结果字段为 `hd95_pixels`，不是毫米。全空或未定义的指标记为 JSON `null`。

## SPIDER 完整体数据评估

SPIDER 的正式比较使用 3D 重建后的完整病例。新的频率实验完成后，可按保存的组名及选模规则评估：

```bash
cd /home/zhp/ftp_next
/home/zhp/.conda/envs/mamba/bin/python evaluate_volumes.py \
  --run-dir /home/zhp/ftp_next/outputs/frequency_three_seed42_run1 \
  --output-dir /home/zhp/ftp_next/outputs/volume_evaluation_new \
  --checkpoint-name best.pt --device cuda:0 --metric-workers 4
```

默认评估完整 valid/test，按病例编号和数字切片序号分组，缺片/重复即报错。先验证 PNG 标签经原预处理逆变换后与 MHA 逐体素一致，再对预测采用同一变换；保存的 3D MHA 也校验数组和空间信息。S1、S2、教师与概率集成分开报告，相同 checkpoint 子网络状态可复用结果，并记录来源。

主指标为逐病例、GT 存在的前景类别均值，再对病例等权平均；每类也分别统计。Dice、IoU 衡量 3D 重叠；教师另报 HD95/ASD（mm，按 MHA spacing 的 NumPy 轴顺序计算）。ASD 沿用原项目的预测表面到 GT 表面的单向定义。漏检类别 Dice/IoU=0，距离为未定义，并报告有效距离病例数及漏检数；GT 不存在时的假阳性计数、包含这些假阳性的补充重叠指标也会保存。

`results.json`、逐类别/病例 CSV 与完整 3D 预测位于输出目录。`--audit-only` 仅检查重建，`--networks teacher` 可仅评估最终教师，`--arms` 可指定已完成的部分组；默认固定 `last.pt`，`best.pt` 必须来自完整 valid 3D 选模。仍兼容旧三组 pilot 协议。

2026-10-16 核查发现 SPIDER_126/192/195 的既有 MHA spacing 轴顺序与原始 mask 不一致。`correct_volume_geometry.py` 接受经过逐体素匹配验证的原始 mask/轴变换记录，另写 `results_geometry_corrected.json`、校正后 CSV 和 MHA，保留原始记录与数据。Dice/IoU 不受间距校正影响。

对新的评估目录运行 `correct_volume_geometry.py --evaluation-dir <评估目录> --trace data/spider_geometry_trace.json`，得到正式采用的校正后距离指标与预测 MHA。旧评估分数已清理，不把切片诊断分数当作 3D 性能。

`use_apgm`、`use_psgm`、`use_frequency` 可以在 YAML 中单独关闭以便后续消融；默认全部开启。关闭提示模块仍保留两种输入视图和 MT 训练，因此不等同于独立的标准单视图 U-Net 基线。

## 已完成验证（2026-10-08）

```bash
cd /home/zhp/ftp_next
/home/zhp/.conda/envs/mamba/bin/python -m unittest discover -s tests -v
```

9 项测试通过：原 APGM 类定义一致性、原 PSGM 输出/输入梯度一致性、学生及提示模块梯度、教师冻结和 EMA、BN buffer 更新、未标注 mask 隔离、未标注 GT 不影响损失、采样/配置/指标，以及无需 BERT 的 checkpoint 重载。源码也通过 Python 编译检查。

真实数据记录位于 `outputs/smoke_real_20261008/`：使用原冻结 BERT、类别向量和完整默认模型，在 CPU 上完成两次优化及 EMA 更新，并对两个 validation 样本完成推理。两步总损失分别约为 4.62407 和 4.76577，均为有限值。这些检查只说明执行链路正常，不构成收敛或精度证据。

当时沙箱内无法访问 CUDA，因此该次记录仅包含 CPU 验证。2026-10-15 已在沙箱外 GPU 完成双学生 batch=2 的两步短测，峰值分配显存约 2GB；完整 batch 的显存与完整训练性能仍需单独验证。
