# 重构验证与启动记录（2026-10-21）

## Material Passport

- 任务：完成频率双学生重构并启动效果对照。
- 验证状态：软件测试通过，真实数据完整链路通过；模型提升尚未验证。
- 用户最新要求：只确认训练能够启动，后续结果由用户提供，不持续监控训练。

## 已完成验证

全部46项新旧测试在现有GPU环境通过，无跳过；日志：`outputs/refactor_audit_20261021/final_tests_gpu.log`。Python编译检查通过。

新增16项测试包含：有符号Haar重建、奇数尺寸、独立学生/EMA、teacher冻结、伪标签停止梯度、空mask可反传、背景筛选与全像素归一、CutMix图像/标签对齐、未标注GT隔离、增强随机流隔离、CPU/GPU严格负对照、交叉监督确实改变S1、完整3D评估、checkpoint重载、epoch中途暂停/恢复逐元素一致、崩溃日志尾部归档。

真实SPIDER检查位于`outputs/freqssl_v3_smoke_20261021_r3/`：完整224输入、batch=8、双学生8次优化，随后评估全部21例valid病例及6种网络输出；完成约41.7秒。这个极短过程不是收敛实验，不用其Dice宣称有效。

GPU两步合成数据检查的峰值分配显存约1669.8 MiB；它不是完整训练或所有配置的显存上限。

开发过程中发现并修复了三个兼容问题：PyTorch2.0不支持的确定性CUDA NLL归约、uint16 PNG直接转tensor、CUDA bincount确定性限制。对应采用等价log_softmax/gather、转换numpy dtype、逐类相等归约；未通过关闭严格确定性解决。早期失败短测目录保留用于追溯，最终测试日志及r3结果为有效验证记录。

## 已启动的配对筛查

- 启动时间：2026-10-21 10:35（Asia/Shanghai）。
- GPU：0；调度进程PID：151718。
- 输出根目录：`outputs/freqssl_v3_screening_20261021/`。
- 顺序：`mean_teacher` → `dual_independent` → `dual_cps` → `dual_frequency`。
- 每组：seed=42、6000步、batch=8（4标注+4未标注）、224输入、1496/2236训练切片。
- 学习率3e-4，前600步监督预热，之后1200步线性增加未标注损失。
- 每500步在全部21例valid病例上评估，按teacher1的3D Dice选模；独立双学生负对照若不一致即停止。
- 每组时间上限14400秒；失败不自动重试。这里只做单种子筛查，不等同于多种子确认。
- 实际执行冻结源码位于运行目录的`code/`，`launch_source_hashes.json`记录hash；工作区后续编辑不改变排队代码。

`launch.json`记录PID/命令；`console.log`为启动和训练日志；`comparison/results.json`记录四组状态；每组目录保存自己的`status.json`、`metrics.jsonl`、源码和checkpoint。首次启动核对写入`startup_check.json`，之后由用户决定何时反馈结果。

一次性启动核对已完成：10:37（Asia/Shanghai），进程151718存活，首组`seed42/mean_teacher`已完成500次优化，最近训练损失0.404390且所有已记录损失有限。此后未安排持续监控，也未继续检查训练指标。

现有标注切片边界使`SPIDER_069`同时出现在标注与未标注训练池，已在各组`split.json`明确记录；三种数据划分间无病例重叠。此次没有读取test图像或标签。正式效果判断需比较同预算单路主指标、各类覆盖率和固定验证选模结果，集成收益单列。
