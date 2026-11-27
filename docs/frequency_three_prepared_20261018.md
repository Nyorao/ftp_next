# 三组频率实验准备记录

## 2026-10-19 10:04 大批量重启（当前执行设置）

用户要求提高新训练显存占用，已中断两个 batch=4 双学生进程，保留其短程日志并标记 interrupted.json。两组使用相同冻结代码从头重启，batch=34，其中标注/无标注各 17。1496/17=88 步/轮，200 轮共 17600 步；每 5 轮（440 步）验证完整 3D。反馈至少预热 50 轮，双学生 Dice 连续两次均达到 0.70 后启用，随后渐增 20 轮，规则不变。学习率、EMA 系数、严格确定性和网络宽度保持原配置。

新输出为 `outputs/frequency_three_seed42_run1_dual_raw_batch34`（GPU 0）及 `outputs/frequency_three_seed42_run1_dual_frequency_batch34`（GPU 1）。实时进程和运行目录见 parallel_dispatch.json。真实数据三步测试通过，峰值 allocated=11578 MiB、reserved=13916 MiB，记录在 `outputs/frequency_batch_probe/batch34.log`。

单学生继续 batch=4、200 轮。新双学生的 batch、BN 统计及总优化更新次数与单学生不同，不能把对单学生的性能差异全部归因于架构；两组双学生之间批量一致。后续严格三组架构消融需要匹配 batch 的单学生对照。本次未启动额外实验。

以下为历史执行记录；与本段冲突时以本段和最新 plan.json 为准。

## 2026-10-19 并行启动更新

用户授权使用两张空闲 GPU。北京时间 09:52 已启动独立的 `dual_raw`（GPU 0，PID 210911）和 `dual_frequency`（GPU 1，PID 210912）；`anchor_raw` 继续在 GPU 3（PID 114367）训练。每组仍为 200 轮，使用原 `queued_source` 冻结代码及配置，没有改变数据、种子、验证或反馈门槛。

双学生独立运行目录分别为 `outputs/frequency_three_seed42_run1_dual_raw`、`outputs/frequency_three_seed42_run1_dual_frequency`，各自保存 manifest、results、console.log、checkpoint。原目录下对应组名已链接至独立运行目录。当前进程与目录映射见 `outputs/frequency_three_seed42_run1/parallel_dispatch.json`。

防重复机制：原串行进程完成并保存 `anchor_raw/result.json` 后，会在创建已存在的 `dual_raw` 路径时触发 FileExistsError，早于构建下一组模型。届时原 queue/results 可能标为 failed，这是预期的串行调度终止；必须结合 anchor 的 result.json 与两个独立运行目录判断实际实验状态，不能把该调度错误当成单学生训练失败。不要重启原串行队列。

以下保留最初排队记录，当前调度以上述并行更新为准。

状态（启动时）：queued_waiting_for_idle_gpu。用户已明确授权自动排队、提前训练。新队列 PID 为 2070825，输出目录为 `outputs/frequency_three_seed42_run1`；实时状态以该目录的 `queue.json` 为准。启动时四张 GPU 均有计算进程，尚未开始模型训练。旧队列保持取消状态。

| 顺序 | 组名 | S1 输入 | S2 输入 | S2→S1 反馈 |
| --- | --- | --- | --- | --- |
| 1 | anchor_raw | 原图 | 无 | 无 |
| 2 | dual_raw | 原图 | 原图 | 有 |
| 3 | dual_frequency | 原图+LF | 原图+HF | 有 |

三组均关闭 APGM、PSGM、提示监督及 BERT 加载；保留 BERT 文件。教师只由 S1 EMA 更新，两学生不共享参数。为匹配视觉网络输入通道数，原图组内部使用 RAW+RAW 两通道，不读取 LF/HF 内容进入网络。

固定设置：seed=42；完整既有训练池 1496 张标注、2236 张无标注；batch=4（2+2）；224 输入；每组 **200 个完整 epoch**；相同优化器、初始化配对规则与数据增强。每个 epoch 定义为标注池遍历一次：1496/2=748 步，总计 **149600 步**。无标注样本从完整 2236 张池中随机抽取，每轮不保证恰好遍历全部无标注样本。入口从 epoch 数和实际 sampler 长度计算总步数，不受旧 10000 步上限截断。

反馈方向按用户确认继续保持 **S2→S1**。前至少 50 轮关闭学生间反馈，学生使用有标注损失及教师一致性学习；其后需 S1、S2 的完整 3D 验证 Dice 均达到 0.70，并在预热后的连续两次验证中满足，才允许反馈。每 5 轮验证一次，因此最早第 55 轮结束确认、第 56 轮开始反馈；若不达标就一直关闭，不以轮次替代能力检查。70% 是预设的工程启动门槛，不能保证每个稀有类别已学好，逐类别结果也会保存。

反馈从实际启用时间起用 20 轮渐增至最大权重 0.05；置信度阈值 0.7、margin 0.05。两组双学生规则完全相同，但可能在不同轮次达标，报告必须披露实际启用轮次。教师一致性在前 40 轮渐增。单学生的反馈权重始终为 0。

每 **5 个 epoch（3740 步）**评估完整验证集 21 个 3D 体的 Dice/IoU，教师、S1、S2、集成分别保存；按完整验证集教师 Dice 保存 best.pt，另存固定预算 last.pt。反馈门控状态、两个学生的验证分数和实际开启轮次随日志及 checkpoint 保存。测试集不参与门控或选模；正式测试时重建完整 3D，并使用已核实的 spacing 修正计算距离指标。

组 2 对比组 1 检验双学生及反馈的增量；组 3 对比组 2 检验 LF/HF 输入方案的整体增量。由于组 3 同时改变两路输入，这三组不能进一步把增益单独归因于 S2 的高频信息。教师和学生集成分别报告。

配置位于 `configs/frequency_three/`，`plan.json` 记录状态、顺序与预算。默认比较入口已改为这三组；其他历史消融只能显式指定，不会默认加入。

已在后台执行以下自动队列命令（记录用途，请勿重复启动）：

```bash
/home/zhp/.conda/envs/mamba/bin/python -u experiments/queue_frequency.py \
  --epochs 200 --output-dir outputs/frequency_three_seed42_run1
```

队列每 60 秒检查 GPU；连续两次满足无计算进程、已用显存不超过 256 MiB、空闲显存至少 8000 MiB、利用率不超过 5% 时自动选择 GPU，并在同一张卡上依次完成三组。最长等待 48 小时，过期则停止等待；训练失败时记录 failed，不自动重跑。无需再次通知启动。运行代码与基础配置已冻结在 `queued_source/`，哈希见 `queued_source_hashes.json`；启动信息见 `launch.json`，队列日志为 `queue_console.log`，训练开始后日志为 `console.log`。

排队前已通过 4 项纯调度/损失检查：完整 epoch 总步数、两个学生共同达标、连续达标计数重置、未通过能力门控时即使到第 199 轮仍不启用反馈。上述检查没有模型训练或优化器更新；正式训练将由当前队列在 GPU 空闲后启动。
