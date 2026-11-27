# V3双学生100/200轮训练（2026-10-26）

用户选择延长原频率双学生、伪标签Dice单项、区域关系单项三组。沿用不新增单学生训练的要求；不延长此前退化的两项组合组。

## 轮数与比较范围

当前双流采样器以完整遍历标注池定义一个epoch：1496张标注切片、每批4张标注，共 **374步/轮**；每批另采样4张未标注切片。它不是把全部3732张训练切片各遍历一次。

| 节点 | 累计更新数 |
|---|---:|
| 之前的筛查 | 6000步，约16.04轮 |
| 100轮 | 37400步 |
| 200轮 | 74800步 |

三组均使用seed42、相同数据划分和S1初始化、batch8、224²、AdamW基础学习率0.0003。交叉伪标签CE、前景/背景阈值、Haar输入、噪声/CutMix、EMA、预热600步与渐增1200步保持原设置。

| 组名 | 伪标签Dice权重 | 区域关系权重 |
|---|---:|---:|
| `dual_base` | 0 | 0 |
| `dual_dice` | 0.1 | 0 |
| `dual_region` | 0 | 1.0 |

现有单学生与旧双学生仅训练6000步，是历史参考，不能作为等训练预算对照来证明双学生架构收益。三组长训练之间才具有匹配的更新预算。启动凭据明确记录这一差别，不绕过原有配对实验的预算校验。

## 连续日程与两个节点

旧6000步训练的多项式学习率已衰减至接近零；直接把总步数改到74800会改变原日程。本轮从相同种子重新初始化，预先固定完整200轮日程：

`lr(step) = 0.0003 × (1 − step / 74800)^0.9`。

第一段运行至100轮，保存完整学生/教师、AdamW、EMA计数和RNG，然后按相同配置恢复至200轮。恢复时总预算仍是74800，不重置优化器、学习率或伪标签预热。100轮是200轮日程中的中间节点，不是另一次独立的“100轮末尾学习率归零”实验。

每5轮（1870步）在全部21例原生3D valid上评估，按teacher1 Dice选模。训练不读取test图像/标签，不自动按test分数调整训练。

每组保存：

```text
<group>/
  protocol.json
  results.json
  training/
    config.json
    metrics.jsonl
    status.json
    best.pt                   # 截至当前的全局valid最佳
    last.pt                   # 继续训练使用的完整状态
  epochs/
    0100/
      at_epoch.pt             # 恰好100轮的模型
      best_through_epoch.pt   # 前100轮内按valid选出的最佳模型
      at_epoch_valid.json
      best_valid.json
      summary.json            # 步数、分数、checkpoint SHA256
    0200/
      ...                     # 同上，200轮节点
```

节点目录原子写入，保留两个节点的完整状态；200轮后的最佳模型不会覆盖100轮的最佳模型。一次调用的时间上限24小时，两段分别计时。出现失败时保留最近成功checkpoint，不能把失败视为完成。

## 启动与恢复

```bash
/home/zhp/.conda/envs/mamba/bin/python experiments/launch_long_v3.py \
  --config configs/v3/long_200epochs.yaml \
  --arms dual_base dual_dice dual_region --gpus 0 1 2 \
  --epochs 100 200 --eval-every-epochs 5 \
  --output-dir outputs/freqssl_long_20261026
```

启动器先检查三张GPU是否空闲，再冻结源码与各组配置，从同一冻结代码目录启动三个独立进程。`launch.json`保存PID、GPU、命令和历史参考分数；每组日志在同级`<group>.log`。仅确认启动，不配置持续监控。

系统盘检查时仅剩约2.9GB，因此本轮输出实际放在 `/data1/zhp/ftp_next_runs/freqssl_long_20261026`（启动时约3.2TiB空闲），仓库 `outputs/freqssl_long_20261026` 为符号链接。新生成的短测目录也完整迁移到该数据盘并保留原入口；旧实验没有移动。启动器现在要求每组至少预留2GiB空间。

2026-10-26 16:45（Asia/Shanghai）已启动并确认每组至少125步：

| 组名 | GPU | PID |
|---|---:|---:|
| `dual_base` | 0 | 696813 |
| `dual_dice` | 1 | 696814 |
| `dual_region` | 2 | 696815 |

三组S1初始化哈希一致，源码快照与工作区一致；正式配置为74800步、每1870步验证，权重与表格匹配。启动核验凭据见 `outputs/freqssl_long_checks_20261026/startup_verification.json`。此后不持续监控训练，尚无100/200轮性能结论。

如某组需要恢复，使用该次运行的冻结源码和配置，例如：

```bash
/home/zhp/.conda/envs/mamba/bin/python \
  outputs/freqssl_long_20261026/code/experiments/train_long_v3.py \
  --config outputs/freqssl_long_20261026/dual_region.yaml \
  --output-dir outputs/freqssl_long_20261026/dual_region \
  --device cuda:2 --epochs 100 200 --eval-every-epochs 5 --resume
```

恢复前应先确认该组原进程已经退出，避免同时写入同一个训练目录。

节点完成后可单独评估已固定的checkpoint。例如100轮最佳模型：

```bash
/home/zhp/.conda/envs/mamba/bin/python \
  outputs/freqssl_long_20261026/code/evaluate.py \
  --checkpoint outputs/freqssl_long_20261026/dual_region/epochs/0100/best_through_epoch.pt \
  --split test --network teacher1 --device cuda:2 \
  --output-dir outputs/long_region_epoch100_test
```

## 验证与证据

`outputs/freqssl_long_checks_20261026/unit_tests.log`：65项测试通过，包括CPU/CUDA分段训练与不中断训练的全部指标和参数逐值相同、学习率跨节点继续下降、节点写入后中断恢复、后续更好的checkpoint不覆盖此前节点快照。原模型损失和推理代码未改动。

真实数据短测使用正式backbone、batch与224²输入，仅缩小训练采样池到8张标注/8张未标注，做2＋2次更新和两次完整21例valid；它检查分段恢复链路，不用于判断性能。证据写入`outputs/freqssl_long_checks_20261026/real_data_staged_smoke.json`。

## Material Passport

本轮为已授权的训练时长实验。材料来自本地代码、数据划分和既有实验记录，无外部上传。保存随机种子、日程、冻结源码、原生3D验证及节点checkpoint哈希。历史单学生预算不同，100轮节点采用200轮日程，均明确披露。长训练效果尚待结果，不把启动或软件测试通过视为性能提升。

## 2026-10-27完成与测试

三组训练均已完成74800步。完整测试报告见 `outputs/freqssl_long_test_20261027/report.md`，固定checkpoint与源码哈希见同目录 `manifest.json`。测试采用与历史结果完全一致的21例、481切片；主网络teacher1，按valid选模，无测试更新。

| 组 | valid最佳轮次 | 最佳模型test Dice % | 第100轮test % | 第200轮test % |
|---|---:|---:|---:|---:|
| 原频率双学生 | 95 | 78.322 | 77.602 | 76.834 |
| Dice单项 | 75 | 78.536 | 77.557 | 75.753 |
| 区域关系单项 | 95 | 78.141 | 78.446 | 76.659 |

100/200轮的best-through checkpoint逐组哈希相同，后100轮没有刷新valid最佳。Dice单项相对同预算原方案提高0.214个百分点，区域关系降低0.181个百分点；三组最佳模型相对各自6000步模型分别降低0.312、0.155、0.608个百分点，均未超过历史单学生79.244%。末尾模型从100到200轮均退化，提示延长训练没有改善泛化；仅凭这些分数不能确定是过拟合、伪标签误差积累还是其他原因。仍为单种子结果；历史单学生预算不同。本次未启动任何新训练。
