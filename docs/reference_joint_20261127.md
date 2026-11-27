# 前后片 + 病例参考片 KV-pos 联合实验

2026-11-27 实现，**尚未在 GPU 上训练**。本文只说明结构、配对设计、判定规则和命令；通过 CPU 测试只说明链路正确，不说明联合有效。

## 1. 先说缺陷和前提

1. **仓库里没有“前后片 KV-pos”的旧实现或结果。** 本次的前后片分支是按“前后片也走 KV-pos”新写的，可能与之前测过的版本不同，所以加了单独的 NB-pos 组，联合结果只和本仓库同一实现的单分支比较。
2. **仓库里看不到 KV-pos 的提升数字。** 旧 README 写 v2“只做了实验准备，尚未正式训练”（`docs/reference_conditioning_20261125.md` 只说脚本默认不启动训练）；但 `experiments/diagnose_kv_position_followup.py` 的说明是“three completed KV-pos checkpoints”，并按 seed42/43/44 读取第 6000 步 `KV-pos/last.pt`，说明写这个脚本时三种子 KV-pos 已训练完成。结果在 `outputs/`（不在仓库），提升幅度这里无法核对。
3. **病例参考片不是固定的中间索引片。** 沿用 KV-pos 原规则：训练时从“冻结 A 预测前景面积 > 0 且 ≥ 本病例最大值 10%”的片中均匀随机抽（尽量排除目标片），推理时取冻结 A 预测椎管面积最大的片，并列取靠近中点者。这与 B 组的 `floor((N-1)/2)` 不同。
4. **位置键的数学后果（已用代码和测试核对）。** KV-pos 的 key 只含 SI 位置编码，注意力 logit 只取决于查询内容和 SI 坐标差，同一 SI 高度上所有 AP 位置的权重相同。所以它实际是“按 SI 高度汇总参考片掩膜特征”，不是逐像素对应。前后两片的 key 完全相同，放进同一个 softmax 后两片权重相等，前后片分支**等价于两片 V 的注意力加权平均**；交换两片输出不变（`test_absent_slots_patients_and_slot_order`）。
5. **训练和部署时的参考质量不一致。** 训练时的参考概率是冻结 A 在训练集上的预测，A 训练时见过其中的标注片，比验证/部署时更准。三组都有这个问题；前后片与目标片高度相关，这个问题可能更明显。
6. **容量混杂。** 每个分支 +610,994 参数（+10.74%），联合 +21.48%。联合若更好，可能部分来自容量；本轮没有同容量对照。
7. **只在开发 valid（21 例）上比较，不读 test。** 判定门槛是工程门槛，不是显著性检验。

## 2. 结构

目标三联片 `[i−1, i, i+1]`、共享编码器和三路 CRF 与 A 完全相同。在 stride 8/16 的 CRF 输出 `Fcrf` 上加两个**独立**的参考残差：

```
Fout = Fcrf + tanh(g_case) · Conv1x1(CA_case) + tanh(g_nb) · Conv1x1(CA_nb)
```

| 分支 | 参考片 | K | V | 参数 |
|---|---|---|---|---|
| 病例（原 KV-pos） | 1 张，规则见 1.3 | 仅 SI 位置编码 | 独立 `MaskEncoder`(参考原图 + 冻结 A 的 20 类概率) | `mask_encoder.*`、`reference_adapters.*`，与原 KV-pos 同名同初始化 |
| 前后片（新） | i−1、i+1 两张 | 仅 SI 位置编码（不读图像） | 另一个独立 `MaskEncoder` | `neighbor_mask_encoder.*`、`neighbor_adapters.*` |

- 两分支各自 4 头、128 维、窗口 stride8 ±2 行 / stride16 ±1 行，门零初始化。两个门都为 0 时三组输出与 A 逐位相同（测试核对）。
- 前后片的两张按 AP 方向拼成一个 key 集合，同一样本共享一个 softmax；不同样本、不同病例的 key 不混在一起。
- 病例两端缺少 i−1 或 i+1 时，该槽有效性为 0，不读、不复制目标片或另一侧邻片；单片病例前后片残差严格为 0。注意 A 的三联片在端点复制目标片，这里不复制。
- 位置键分支不需要参考片的编码器特征，所以前后片分支不跑共享编码器；原 KV-pos 分支仍保留原来的 `content*0 + PE` 写法以保证逐位兼容。

## 3. 配对设计

- `reference_sources: case` 走的就是原 KV-pos 代码路径。用原分支 PR #1 的代码和新代码各跑同一个 4 步合成训练，学生/教师哈希、每步 loss、验证结果完全一致（2026-11-27，CPU，本地核对，未进 CI）。
- 骨干、CRF、病例分支在三组中初始化相同；前后片分支在 NB-pos 和 JOINT-pos 中初始化相同，且与病例分支不同（测试核对）。新模块的初始化不改变全局随机流。
- 数据：目标片、标签、几何增强、病例参考片抽样在三组中逐位相同（测试核对）。前后片使用目标片同一仿射；强度扰动用新的独立随机流（种子偏移 800000011 / 1000000007），不影响原有随机流。
- CutMix：混入区域用供体病例自己的前后片 bank，与原病例分支同一规则。
- 三组共用同一个冻结 A 教师（按 seed 对应）和同一份概率缓存。

## 4. 实验组与判定

| 组 | `reference_sources` | 说明 |
|---|---|---|
| A | — | 已有结果，不重训 |
| KV-pos | `case` | 在本轮缓存上重跑原 KV-pos；seed42 的重跑也用来核对与 2026-11-25 那次是否完全一致 |
| NB-pos | `neighbors` | 仅前后片分支 |
| JOINT-pos | `case_neighbors` | 两个分支同时加 |

全部 6000 步、224²、batch 8 = 4L + 4U、1496 标注 / 3732 影像、warmup 600 / ramp 1200，除路径、seed、`reference_sources` 和 `max_seconds` 外与原 KV-pos 配置完全相同（测试核对）。`max_seconds` 从 14400 改为 43200 只是安全停止时间，超时会保存可恢复的 `last.pt`，不影响训练。

固定比较第 6000 步 EMA 教师（`valid_step_6000.json` 的 `teacher1`），不用 best 检查点。以下规则在看结果前写入 `configs/plans/reference_joint_20261127.yaml`：

- **联合有增益**：JOINT − (KV-pos 与 NB-pos 中三种子平均 Dice 更高者) 平均主 Dice ≥ +0.1pp，至少 2/3 种子为正，平均 union-present 不降。
- **联合超过 A**：沿用原门槛，平均主 Dice ≥ +0.3pp，至少 2/3 种子为正，平均 union-present 不降。
- 同时报告病例配对 bootstrap 95% 区间（每例先平均种子，再重采样 21 例，10000 次）和条件错号率变化。少于三种子时只标 partial，不给结论。

种子计划：先 seed42 三组（JOINT、NB、KV-pos），再 seed43/44。建议无论 seed42 结果如何都补完 43/44：0.1–0.3pp 级别的差异单种子分不开。若 seed42 的 KV-pos 重跑与旧结果完全一致，43/44 的 KV-pos 可以直接用旧结果（汇总脚本会自动回退并注明），省 2 次训练。

## 5. 开销

单目标训练前向（Conv/Linear + 注意力矩阵乘，不含归一化、softmax、插值和反向；不是总 FLOPs）：

| 组 | 参数/学生 | 相对 A 的计数 MAC |
|---|---:|---:|
| A | 5,688,640 | 1.000× |
| KV-pos | 6,299,634 | 1.184× |
| NB-pos | 6,299,634 | 1.036× |
| JOINT-pos | 6,910,628 | 1.220× |

KV-pos 训练时仍对参考图跑一次共享编码器（K 内容乘 0 也要算），所以比 NB-pos 贵。部署三组都要先用冻结 A 预测整卷（需要两套权重）；前后片 V 每卷每片只算一次。

## 6. 命令

服务器上需要（不在 git 里）：`data/spider_image_geometry_manifest.json`；三个种子的 A 目录 `outputs/case_reference_20261112/screen/A/`、`outputs/case_reference_followup_20261116/training/seed43/A/`、`.../seed44/A/`（至少 `last.pt`、`split.json`、`config.json`、`result.json`、`valid_step_6000.json`）；若要和旧 KV-pos 对照，再加 `outputs/reference_conditioning_20261125/training/seed*/KV-pos/` 下同样几个文件。路径都相对仓库根目录，数据根目录用 `SPIDER_PNG_ROOT` 指定。

```sh
git fetch origin && git checkout claude/project-thread-tmkasx && git pull
export SPIDER_PNG_ROOT=/你的/slice_png_order      # 与默认路径相同则不用设

# 0. 测试（GPU 机器上跑，CUDA 测试会自动执行）
python -m pytest -q tests/test_reference_joint.py tests/test_reference_conditioning.py

# 1. 每个 seed 建一次冻结 A 概率缓存（约 8 GB），三组共用
python experiments/build_reference_cache.py --config configs/v3/reference_joint/seed42/JOINT-pos.yaml

# 2. 冒烟：8 步 + 完整 21 例验证
python train.py --config configs/v3/reference_joint/seed42/JOINT-pos.yaml --smoke-test --output-dir outputs/reference_joint_smoke/JOINT-pos
python train.py --config configs/v3/reference_joint/seed42/NB-pos.yaml --smoke-test --output-dir outputs/reference_joint_smoke/NB-pos

# 3. 正式（可分到不同 GPU 并行，CUDA_VISIBLE_DEVICES=0/1/2）
mkdir -p outputs/reference_joint_20261127/logs
nohup python train.py --config configs/v3/reference_joint/seed42/JOINT-pos.yaml > outputs/reference_joint_20261127/logs/seed42_JOINT-pos.log 2>&1 &
nohup python train.py --config configs/v3/reference_joint/seed42/NB-pos.yaml    > outputs/reference_joint_20261127/logs/seed42_NB-pos.log 2>&1 &
nohup python train.py --config configs/v3/reference_joint/seed42/KV-pos.yaml    > outputs/reference_joint_20261127/logs/seed42_KV-pos.log 2>&1 &

# 4. 汇总（随时可跑，缺的种子标 partial）
python experiments/summarize_reference_joint.py
```

seed43/44 把路径中的 `seed42` 换成 `seed43`/`seed44`，从第 1 步开始。

注意：

- 日志不要放进训练输出目录；`train.py` 发现输出目录非空会拒绝启动。
- 中断后用 `python train.py --resume outputs/reference_joint_20261127/training/seed42/JOINT-pos/last.pt` 继续。训练期间不要 `git pull`，恢复时会核对源码哈希。
- 冒烟测试的 1200 秒上限包含数据和几何校验时间；若只报 `TimeoutError`，先看是不是校验本身耗时。
- seed43/44 的 A 权重哈希不在仓库里，配置中留空；缓存清单会记录实际哈希，汇总脚本会检查同一 seed 各组用的是同一个教师。

产物：`outputs/reference_joint_20261127/cache/seed*/`（缓存，可在三组都跑完后删除）、`training/seed*/<组>/`（`last.pt`、`valid_step_*.json`、`metrics.jsonl`，日志中有 `reference_neighbor_gates`）、`summary/summary.{md,json}` 与 `per_case.csv`。回传 `summary.md` 即可。

## 7. 已核 / 未核

- 已核（CPU，Python 3.13，PyTorch 2.15）：本文件提到的所有“测试核对”项，见 `tests/test_reference_joint.py`；原 KV-pos 路径逐位兼容；整卷缓存推理与直接前向一致（含微批边界与端点）；训练、断点恢复、检查点加载、无缓存部署。
- 未核：CUDA 上的确定性反向和显存/耗时（已有 CUDA 测试，需在服务器跑）；任何性能结论。
