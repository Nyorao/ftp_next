# 无提示模块的频率双学生：文献依据与实验方案

日期：2026-10-18。后续用户要求已将实验收敛为三组、每组 200 个完整 epoch；保留单向反馈，并在两学生预热及 3D 验证达标后才启用。原自动队列已停止；用户现已授权提前训练，新的三组自动队列已启动，实时状态见 `outputs/frequency_three_seed42_run1/queue.json`。最新执行约定见 [三组准备记录](frequency_three_prepared_20261018.md)。本文保留文献依据及此前五组、10000 步方案讨论，后者不再是执行计划。

## 本轮范围

按用户要求，删除 2026-10-15 双学生 pilot、probe 及 2026-10-16 对应 3D 评估输出和结果报告。清理清单见 `training_cleanup_20261018.json`；原始数据、原 FTP 代码、BERT 文件未修改。三例 spacing 的原始 mask 匹配证据独立保存在 `data/spider_geometry_trace.json`、`data/spider_geometry_audit.json`，不依赖已删除的模型结果。

本轮配置 `configs/frequency_dual.yaml` 关闭 APGM/PSGM。关闭时不加载 BERT/类别向量；checkpoint 中仅保留不参与计算的零值形状 buffer，保持视觉瓶颈宽度和加载兼容。分支仅解码一次，防止无提示模块时重复更新解码器 BN。提示监督及其权重均为零。旧提示模块实现可通过旧配置重新启用。

当前 backbone 仍是随机初始化的 U-Net；Mean Teacher 是训练机制。两学生独立初始化、不共享参数。输入使用原图与既有小波低频/高频图的两通道拼接，不是只给模型纯高频图。

```text
RAW + LF → S1 U-Net ── EMA ──→ 单教师 T
                 ↑                │
           可选 S2 反馈             ├─ 一致性目标 → S1
RAW + HF → S2 U-Net                └─ 一致性目标 → S2
```

教师只跟随 S1。无反馈时 S2 无法改变教师，必须明确区分最终单教师收益与双学生集成收益。

## 相关论文核查

这是针对当前设计的快速检索，不是系统综述；论文报告的收益不等于本地复现。

| 论文 | 可支持的设计依据 | 与当前方案的区别 |
| --- | --- | --- |
| XNet，ICCV 2023 | 小波 LF/HF 双分支、特征融合和双输出一致性用于有监督及半监督生物医学分割；作者代码列出 GlaS、CREMI、LA、LiTS 比较 | 有跨分支特征融合；不是当前两套独立 U-Net 加一个 EMA 教师的直接验证 |
| XNet v2，2024 arXiv 版本 | 讨论纯 HF 信息不足导致的退化；增加原图和频率互补融合，在三种 2D、两种 3D 数据上实验 | 三个子网络及图像/特征融合，不是两学生一教师；支持保留原图，但不保证我们简单拼接有效 |
| β-FFT，CVPR 2025 | 频域低频交换/非线性插值与差异化协同训练，研究双网络同质化；作者提供 ACDC、PROMISE12、MS-CMRSeg 代码 | 不是固定 LF 专家/HF 专家的分工；还有额外训练和插值正则，不能把整体收益单归于频率输入 |
| SDCL，2024 arXiv 版本 | 两学生、一 EMA 教师；教师跟随学生 A，用预测分歧指导纠错；在 Pancreas、LA、ACDC 上实验 | 用 UNet/ResUNet 或 VNet/ResVNet 制造差异，并使用 BCP、预热和纠错损失；不验证频率分工 |

来源与核查位置：

1. Zhou et al., **XNet: Wavelet-Based Low and High Frequency Fusion Networks for Fully- and Semi-Supervised Semantic Segmentation of Biomedical Images**，ICCV 2023：[论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_XNet_Wavelet-Based_Low_and_High_Frequency_Fusion_Networks_for_Fully-_ICCV_2023_paper.pdf)、[作者代码与架构/实验说明](https://github.com/Yanfeng-Zhou/XNet)。本次 CVF 直读返回 403，使用已索引论文内容和作者仓库交叉核查，不声称完整通读 PDF。
2. Zhou et al., **XNet v2: Fewer Limitations, Better Results and Greater Universality**：[原文](https://arxiv.org/html/2409.00947v1)。重点核查 II-A、II-B、Table I；Table I 的半监督 ISIC-2017 Dice 从 LF+HF 的 83.16 到 LF+HF+Raw 的 83.66，属于作者特定设置，不能外推为 SPIDER 预期收益。
3. Hu et al., **β-FFT: Nonlinear Interpolation and Differentiated Training Strategies for Semi-Supervised Medical Image Segmentation**，CVPR 2025：[论文页面](https://openaccess.thecvf.com/content/CVPR2025/html/Hu_beta-FFT_Nonlinear_Interpolation_and_Differentiated_Training_Strategies_for_Semi-Supervised_Medical_CVPR_2025_paper.html)、[作者代码](https://github.com/Xi-Mu-Yu/beta-FFT)。CVF 直读返回 403，方法和评估范围核对了搜索索引原文与作者 README；不将摘要中的整体提升解释为单因素消融结论。
4. Song and Wang, **SDCL: Students Discrepancy-Informed Correction Learning for Semi-supervised Medical Image Segmentation**：[原文](https://arxiv.org/html/2409.16728v1)。重点核查 2.2：教师 EMA 只由学生 A 更新；与当前参数更新关系相近，但学生差异来源和纠错目标不同。此处按核查的 arXiv 版本标识，不推断后续发表状态。

结论：频率双分支和一教师双学生分别有明确先例，但本次检索尚不能确认“当前 RAW+LF/RAW+HF、纯 U-Net、教师仅 EMA(S1)、筛选 S2→S1”组合已在 SPIDER 上得到验证。简单组合这些要素也不足以主张新颖性。

## 配对消融

所有组禁用 APGM/PSGM，保持数据、初始化种子、优化器、训练更新数和主推理网络预算相同。

| 组名 | S1 输入 | S2 输入 | S2→S1 | 用途 |
| --- | --- | --- | --- | --- |
| anchor_lf | RAW+LF | 无 | 无 | LF 单学生基线 |
| dual_frequency_no_feedback | RAW+LF | RAW+HF | 关闭 | 严格负对照及集成诊断；S1/T 应与 anchor_lf 完全一致 |
| dual_same_view | RAW+LF | RAW+LF | 开启 | 同容量、同协作的非互补视图对照 |
| dual_frequency | RAW+LF | RAW+HF | 开启 | 待验证方案 |
| anchor_raw | RAW+RAW | 无 | 无 | 同输入通道数的原图单学生基线 |

主要比较：`dual_frequency − dual_same_view` 隔离 S2 频率视图的增量作用；`dual_frequency − dual_frequency_no_feedback` 测反馈作用；`anchor_lf − anchor_raw` 测 LF 额外通道作用。不同初始化的同视图学生也可能互补，因此只比较双学生与单学生是不够的。

## 训练及评估约定

- 使用原有完整训练池：1496 张标注、2236 张无标注，未标注 GT 不读取。沿用既有切片标注规则，不能把它称为患者级标注比例。
- 首轮 seed=42，每组 10000 更新，batch=4（2 标注+2 无标注），224 输入，视觉网络从零训练。该预算用于学习曲线诊断，不预先声称收敛。
- 前 20% 更新关闭 S2 反馈，随后 20% 渐增；最大权重 0.05、置信度 0.7、margin 0.05。反馈仍为既有筛选软 KL；本轮不叠加新的纠错模块。所有反馈组设置一致。
- 教师一致性前 20% 更新渐增；关闭提示监督，保留 CE/Dice 和教师一致性的原有定义。记录反馈像素覆盖、分支监督损失与总损失。
- 每 2000 更新评估完整 valid 的 21 个 3D 病例，固定 last.pt 与按完整 valid 教师 Dice 选取的 best.pt 分别保存。训练中评估 Dice/IoU；末次同时评估 S1、S2、集成。测试集只检查文件名划分，既不读取图像/标签，也不用于调参。
- 正式核查测试集时重建完整 3D，应用已核实的 spacing 修正，报告 Dice、IoU、HD95、单向 ASD、各类别漏检与距离有效数。`evaluate_volumes.py` 已支持新组名、已完成的部分组及 `--checkpoint-name best.pt/last.pt`，默认仍为固定 last.pt；随后用 `correct_volume_geometry.py --trace data/spider_geometry_trace.json` 另存距离校正结果。选模依据及实际组名写入评估 manifest。
- 第一轮只有一个种子。若观察到增益，再固定设置做多种子和同 GPU 时间预算比较；不因一次均值提高即认定有效。

## 执行

直接运行（已有独占空闲 GPU 时）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zhp/.conda/envs/mamba/bin/python experiments/compare_frequency.py \
  --device cuda:0 --output-dir outputs/frequency_comparison_new
```

等待空闲 GPU：

```bash
/home/zhp/.conda/envs/mamba/bin/python experiments/queue_frequency.py \
  --output-dir outputs/frequency_ablation_20261018_seed42
```

队列固定源码快照，每分钟检查一次；同一 GPU 连续两次满足无计算进程、已用显存不超过 256 MiB、空闲至少 8 GiB、利用率不超过 5% 才启动。单进程顺序执行五组，不停止其他任务；48 小时仍未空闲则记为过期。`queue.json` 区分等待、训练、完成、失败；不会把排队状态写成已训练。检查与启动之间仍存在其他用户抢占设备的时间窗口，不能视为调度器提供的独占锁。

本地验证：26 项测试中 25 项通过、1 项 GPU 测试因沙箱设备不可见跳过；新增测试覆盖无 BERT 依赖、无提示损失、单次 BN 更新、同频视图路由，以及合成完整 3D 训练/选模/负对照。另以真实 SPIDER、224 输入、完整默认视觉宽度在 CPU 完成两步双学生训练，日志位于 `outputs/frequency_smoke_20261018`，这只验证执行链路。

输入核查：均匀抽取 train 32 张、valid 8 张，重新执行原 `wave.py` 的 Haar 小波及 uint8 归一化，40/40 张 LF 和 40/40 张 HF 均与现有文件逐像素相同。证据见 `data/frequency_input_audit_20261018.json`；这是抽查，不声称审计所有频率图。

历史启动（现已按用户要求停止）：2026-10-18 23:28（Asia/Shanghai），队列 PID 1930685，输出目录 `outputs/frequency_ablation_20261018_seed42`。启动核验时四张 GPU 均有其他计算进程，状态为 `waiting_for_idle_gpu`，尚未开始长训练。最新状态以该目录 `queue.json` 为准。
