# Mean Teacher main figure — revision 2

本版按用户对论文主图的反馈重新组织：以学生和 EMA 教师为视觉主体，网络内部结构与融合公式作为局部放大图。上一版保留在 `../context_architecture_20261106/`，未覆盖。

## 交付文件

- `mean_teacher_main_v2.svg`：原生可编辑矢量图，普通标注保留为文本。
- `mean_teacher_main_v2.pdf`：矢量 PDF，适合论文排版。
- `mean_teacher_main_v2.png`：4200 × 2700 高清导出。
- `gen_main_figure.py`：矢量绘图源文件。
- `design_prompt.txt`：本次内置 `image_gen` 图像工具所用构图与细化提示词。
- `design_reference.png`：内置图像工具根据经过核对的排版稿生成的视觉细化参考。最终交付的 PNG、SVG、PDF 均从矢量源导出，保持精确连接与纯色填充。

## 阅读顺序

**(a) Mean Teacher framework** 占主图约三分之二。学生接收标注与未标注强视图，教师接收未标注原视图。学生和教师分别拥有完整的 ContextUNet；学生梯度更新，教师通过 EMA 更新。右侧分别展示中心片 CE/Dice 监督和置信筛选后的伪标签 CE。目标概率与学生未标注图像使用同一个 CutMix 区域 M；图中先混合目标，再做置信筛选，与训练入口的顺序对应。推理只部署教师网络。

**(b) Three-slice ContextUNet** 同时适用于学生和教师各自的网络。三张切片在各自网络内共享编码器权重，学生和教师之间保存不同参数。编码器保留五个尺度；E0、E1 仅使用中心特征，E2、E3、E4 进行三片融合。E4 的融合特征作为解码起点，E3/E2 的融合特征和 E1/E0 的中心特征作为跳跃连接。四级解码后得到中心片 20 类分割。

**(c) Center-residual fusion** 展示三路特征共同计算空间权重与加权和，并以中心特征进行残差融合。三个尺度各自有独立参数；γ 是无区间约束、初始为 0 的可学习标量。空间权重在通道间共享。

主干沿用 32 / 64 / 128 / 256 / 256 通道版本。20 类为背景、9 类椎体编号、椎管与 9 类椎间盘编号。脊柱示意图不代表真实患者输入或模型预测结果。本图不加入压缩候选、提示教师或历史频率双学生模块。

## 科学依据

- `../../freqssl/models.py`：五级 U-Net、四级解码器、教师复制与 EMA 更新。
- `../../freqssl/slice_context.py`：中心残差融合以及两级中心跳跃、三级融合特征。
- `../../freqssl/engine.py`：强视图、对齐 CutMix、标注损失与伪标签损失。
- `../../configs/v3/context_efficiency_reference.yaml`：当前 32 通道配置。
- `../../docs/context_architecture_efficiency_20261106.md`：当前架构说明。

## 再生成

```sh
/home/zhp/.conda/envs/mamba/bin/python figures/mean_teacher_main_20261106_v2/gen_main_figure.py
```

画布与分区底色均为纯白。模块使用淡蓝、薄荷绿、淡紫和浅杏色，无渐变或阴影。SVG/PDF 中不嵌入位图。
