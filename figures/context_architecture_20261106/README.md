# Current ContextUNet architecture figure

本图对应 2026-11-06 工作区中的三片自适应 ContextUNet 主干，采用 32 通道版本，通道序列为 32 / 64 / 128 / 256 / 256。

## 交付文件

- `context_architecture.svg`：可编辑矢量源图，普通标签保留为文本。
- `context_architecture.pdf`：矢量 PDF，适合插入论文。
- `context_architecture.png`：4000 × 2640 高清预览。
- `gen_architecture.py`：确定性矢量绘图脚本。
- `design_prompt.txt`：完整构图提示词；初步视觉设计使用内置 `image_gen` 工具。
- `design_reference.png`：生成式视觉首稿，保留作设计参考。最终交付图由矢量源图导出，连接关系与字体经过单独校正。

## 图中机制

1. **Network**：相邻三个单通道矢状位切片由同一组 2D 编码器权重分别编码。E0、E1 只取中心片跳跃特征；E2、E3、E4 使用独立的中心残差融合模块。四级解码器产生中心片 20 类结果。
2. **Fusion**：三个输入共同进入 Concat / 1×1 Conv / Softmax，得到逐位置三片权重；权重在特征通道间共享。三片加权和经过中心残差输出。每个尺度的可学习标量 γ 初始为零，不是限制在 [0,1] 的门控概率。
3. **Training**：一个学生、一个不接收反向梯度的 EMA 教师。标注中心片使用 0.5 CE + 0.5 Dice；无标注中心片使用置信度筛选后的伪标签 CE。学生未标注图像与教师目标使用同一个 CutMix 区域。EMA 箭头只表示学生参数向教师更新。
4. **Inference**：只部署一个最佳 EMA 网络。体积切片的特征金字塔编码后缓存，供相邻中心片重复使用。缓存是推理机制。

脊柱图标为示意插画，不表示真实患者输入或模型预测结果。20 类包括背景、9 类椎体编号、椎管和 9 类椎间盘编号。图不包含尚未采用的 24 通道压缩候选、历史频率双学生、序数分支或提示教师实验。

## 源码依据

- `../../freqssl/models.py`：五级共享编码器、四级解码器、分割头及 EMA。
- `../../freqssl/slice_context.py`：`CenterResidualFusion`、`ContextUNet._decode_triplets`。
- `../../freqssl/engine.py`：学生强视图、对齐 CutMix、标注及伪标签监督。
- `../../freqssl/context_cache.py`：体积推理的编码预取与特征复用。
- `../../docs/context_architecture_efficiency_20261106.md`：当前架构与计算优化说明。

## 再生成

```sh
/home/zhp/.conda/envs/mamba/bin/python figures/context_architecture_20261106/gen_architecture.py
```

最终背景为纯白色，模块填色为低饱和蓝、薄荷绿、淡紫和浅杏色，无阴影或渐变。SVG/PDF 含原生路径和文字，不是将位图嵌入矢量容器。
