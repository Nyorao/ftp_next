# 邻切片贡献机制：冻结教师诊断协议

## Material Passport

2026-11-05；依据用户批准的机制优先路线。阶段一实现与运行，不改变训练或模型公共接口。
AI辅助实现、文献整理和分析；输入、标注和预测仅在本地处理。没有新增标注或使用test开发。

## 研究问题和候选方向

1. 位置相关的方向贡献：当前切片在扫描中的位置是否与前后邻片的条件贡献相关？
2. 物理距离与上下文范围：更远邻片是否改善预测，这种变化是否依赖真实切片间距？
3. 可靠性与负面贡献：邻片特征差异、纹理代理、输出熵能否指示有害上下文？
4. 互补性与空间对应：双侧联合收益是否超出单侧收益之和，局部位置错配是否与边界错误相关？

前三者分别借鉴 [SegMate §3、表3](https://arxiv.org/html/2602.23903v1)、
[MOSformer §5.3](https://arxiv.org/html/2401.11856v2)、
[CSAM §4.4、表1–2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11349312/)。
第四方向参考 [ASD-Net](https://doi.org/10.1016/j.bspc.2025.107809) 的公开方法片段与
[CSA-Net](https://arxiv.org/html/2405.00130v1)。动态空间权重已有
[CFF](https://arxiv.org/html/2603.24115v1)先例；不主张该机制的新颖性。
[切片重要性偏差研究](https://arxiv.org/html/2505.14105v3)针对RGB预训练，不能直接套用于当前从零训练的共享灰度编码器。

## 冻结设计

- 默认检查点：`spine_slice_context_20261030_r2/confirmation/seed{42,43,44}/adaptive/best.pt`中的单个teacher1。
- 同一完整valid；加载现有最佳checkpoint，不按诊断重新选模。不读train标注或test影像/标注。
- 条件：full；no_previous（中心替上一片）；no_next；center_only；swapped；gap2；gap3。
- 每真实切片每种子仅编码一次，七个条件均沿用原三片融合/解码器和eval微批次；观察hook不替换输出。
- 保持采集序号输入，使用经过影像字节哈希和全部PNG输入等价核验的几何，仅为分析标记物理左右、距离与扫描相对位置。
- 越界索引复制最近端点，记录实际距离、真实片数与是否截断。端点单独汇总；距离诊断同时报告两侧gap3均有效的共同内部区域。
- 每次运行冻结源码、协议、checkpoint/历史指标/几何哈希。目录不可覆盖；失败不自动重试。
- 全输入逐病例比较历史三维Dice和union-present；CPU/CUDA差值容许绝对1e-4（0.01pp），超出停止。运行内配对均以本次full为基准。
- 各种子状态按病例更新；CPU可用--jobs并行独立种子进程，不改变单模型线程数。启动器默认14400秒硬超时，终止本次启动的进程组。每病例JSON/NPZ是完整可检查单元；不支持从失败目录静默续跑。

## 输出和统计

原生三维：保持原评估的argmax、最近邻恢复、转置及体积重建；报告GT-present主Dice、GT-or-prediction-present补充Dice、组织Dice、编号混淆、缺失类别误检对数/体素。

224²诊断：all/foreground/background/vertebra/canal/disc/boundary/interior区域的错误计数、CE、类平衡CE、熵、改变像素和正确/错误转移。
boundary为任意类别转移附近两模型像素的膨胀带，interior为该带外前景；不是物理距离边界指标。
同时区分整例缺失类别与当前片缺失类别的误检像素。报告真实双侧邻片条件下的有害切片比例，
以及各病例/种子事后最优固定范围；后者只描述异质性，不能当作可部署选择器或无偏性能。

令e为错误率：上一片条件贡献=e(no_previous)-e(full)；下一片同理；联合贡献=e(center_only)-e(full)；
交互=e(no_previous)+e(no_next)-e(full)-e(center_only)。正贡献表示该上下文降低了错误。

位置按物理右至左的扫描相对位置划分五个等宽区间；间距预设<3.8、3.8–4.6、≥4.6 mm三组。
每病例先汇总区域计数，不将像素或切片当作独立病例。三个种子先在同病例内平均，再做10000次病例bootstrap（seed20261105）。
相关系数先逐病例计算Spearman，再做同样病例汇总；间距关联是病例间描述性相关。区间不代表种子总体不确定性。

权重：各尺度三片权重、gamma、实际系数[gamma*w-,1-gamma+gamma*w0,gamma*w+]、两侧残差RMS。
质量：归一化影像Laplacian方差、高频MAD，均只是纹理代理。对应：stride4特征同位置余弦差异及半径1网格局部最佳匹配增益，不视作解剖配准。

探索性信号：至少10病例、三个种子齐全、病例bootstrap区间不跨0且至少2/3种子同向。位置采用位置—方向贡献关联；
范围采用远邻三维Dice改善且补充Dice不下降；可靠性采用代理量与贡献的稳定负关联；互补采用交互或边界局部匹配关联。
多个探索性分组未做多重比较校正；信号只排序方法设计候选，最多两个，不直接触发训练或宣称确认。

## 运行

```sh
MPLCONFIGDIR=/tmp/context-atlas-mpl /home/zhp/.conda/envs/mamba/bin/python -m unittest discover -s tests -p test_context_diagnostics.py -v

# 一个病例、一个种子的真实链路短测，不用于方向选择。
MPLCONFIGDIR=/tmp/context-atlas-mpl /home/zhp/.conda/envs/mamba/bin/python experiments/probe_slice_context.py --output-dir outputs/context_atlas_smoke_20261105 --device cpu --seeds 42 --max-cases 1

# 三种子全valid；如果实际有GPU，可显式将device指定为cuda:0。
MPLCONFIGDIR=/tmp/context-atlas-mpl /home/zhp/.conda/envs/mamba/bin/python experiments/probe_slice_context.py --output-dir outputs/context_atlas_20261105 --device cpu --jobs 3
```

`--prepare-only`仅生成冻结包与`command.json`。输出包含manifest/status、几何核验、每种子的历史对齐记录、
逐病例JSON和压缩地图、原生指标、切片区域CSV、summary.json、report.md及静态图。

## 条件性训练路线

诊断完成后最多选择两个有支持的方向，先冻结具体方法与对照。位置比较内容与内容＋位置；
范围先比较近邻三片、远邻三片与五片，再比较自适应与最佳固定范围。可靠性预测只能在有标注训练病例中交叉验证；
验证标签仅用于评估，不得训练门控。保持共享编码、Mean Teacher、中心监督与原标注预算，分离机制并报告额外计算。

筛查seed42/6000步，基准当前adaptive；通过后最佳候选seeds42/43/44各30000步。
确认目标：平均主Dice≥+0.3pp，至少2/3种子正向，平均union-present不下降。此为描述性筛查门槛，不是统计显著性。
此阶段需依赖实际诊断才能定义方法；本工具不提前更改模型接口或生成未经证据支持的训练结果。
