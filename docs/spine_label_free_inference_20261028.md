# 脊柱无标签推理与数据来源核对

`predict_spine.py` 从一幅标量3D影像和一个v3 checkpoint生成保留物理几何的MHA分割，不需要标注文件、数据集目录或类别质心。默认只实例化并加载主教师的一套U-Net。此入口属于工程验证，不构成模型优于基线的证据；OAR-Spine首轮性能结论见[实验报告](../outputs/spine_ordinal_test_20261028/report.md)。

## 来源与患者隔离

对本仓库210个转换MHA进行影像强度指纹检索，再穷举轴置换/翻转并比较完整体素，全部唯一匹配到原始T2影像。168例train、21例valid、21例test分别对应互不重叠的原始患者ID。原始发布明确218个study来自218位患者；文件名由study identifier和序列类型组成，见[原始数据发布](https://zenodo.org/records/10159290)及[数据论文](https://www.nature.com/articles/s41597-024-03090-w)。没有使用掩码匹配患者或推断方向。

188例无需轴置换；另22例采用ZYX置换 `[1,2,0]`，但原转换文件未同步spacing/direction。新清单根据逐体素验证的变换从原始影像重建正确几何，保留原始文件不变。使用转换MHA时，推理入口核对路径、形状及完整影像体素SHA256；内容不匹配即拒绝应用几何修正。

清单：`data/spider_image_geometry_manifest.json`。来源证据位于 `outputs/spine_design_20261027/provenance/` 的 `verified_image_mapping.json`、`image_fingerprints.json`、`patient_split_audit.json`。此结论仅针对已匹配的210例和现有划分。

## 输入与输出

方向来自可信SimpleITK/LPS几何，规范张量为 `(左右切片, 前后, 头尾)`，最后一轴向尾侧增加。允许保留原生轻度倾斜；当轴方向余弦低于0.8时拒绝直接置换，要求先明确重采样流程，不用数组短边猜方向。

每张切片复现原PIL和Matplotlib gray8转换，再作双线性缩放和均值/标准差归一化。模型预测先argmax，再最近邻还原原分辨率，逆置换/翻转后保存带可信spacing、origin、direction的MHA。未使用顺序重编号或连通域后处理。

标签0为背景，1..9为相对椎体编号，10为椎管，11..19为相对椎间盘编号；不能将全部相对编号固定解释成绝对解剖节段。

```sh
# 原始影像：使用可信原始header
/home/zhp/.conda/envs/mamba/bin/python predict_spine.py \
  --checkpoint outputs/spine_ordinal_20261027/hard_peer/best.pt \
  --input /data/zhp/SPIDER/images/100_t2.mha \
  --output outputs/predict_case001.mha --device cuda:0

# 已核对的转换影像：显式使用几何清单
/home/zhp/.conda/envs/mamba/bin/python predict_spine.py \
  --checkpoint outputs/spine_ordinal_20261027/hard_peer/best.pt \
  --input '/data/zhp/dateset_spine copy/spider/mha/train/image/SPIDER_006.mha' \
  --geometry-manifest data/spider_image_geometry_manifest.json \
  --output outputs/predict_case006.mha --device cuda:0
```

支持原始MHA/NIfTI输入，要求可信的3D标量几何。已知错误header的转换影像若不指定清单会报错。默认网络由checkpoint协议决定，也可显式选teacher1/teacher2/student1/student2；不在该入口自动挑选分支或做集成。输出同名JSON保存输入/模型/源码哈希、实际几何、精度设置、库版本和耗时。已有输出或JSON不会覆盖。新入口额外使用Matplotlib复现训练量化方式。

## 完成的验证

| 检查 | 范围 | 结果 |
|---|---|---|
| 影像来源匹配 | 全部210例 | 唯一原始T2来源，无划分间患者交叉 |
| 无标签输入等价 | 全部4631切片：3732 train、418 valid、481 test | 灰度像素和224²归一化张量差异均为0 |
| 物理几何往返 | 48种轴置换/翻转，合成数据 | 物理坐标与SimpleITK一致 |
| 单网络提取 | 单学生、频率双学生、因子化双学生的可用分支 | 与完整训练系统的输出逐值一致 |
| 独立CLI | 数据集路径不存在、仅有影像和checkpoint | 成功推理，防覆盖与哈希核验通过 |
| 真实GPU推理 | 两例train，共38切片，包含轴置换病例 | 原始/转换MHA及PNG的还原标签一致，无标注读取 |

输入等价审计读取了valid/test影像，只用于证明预处理等价，没有模型推理、标签访问或调参。真实推理检查使用既有8步短测checkpoint，不计算分割性能。

## 浮点精度发现

首次真实推理检查保留了一个实际失败：SPIDER_006左右切片逆序改变CUDA末批大小，在旧默认cuDNN TF32设置下，224²网格有18/852,992个argmax像素不同，最大概率差0.000841。模型输入完全一致；CPU不同顺序结果完全一致。关闭TF32后，CUDA概率差降至1.19e-6，观察到的标签差异为0。因此新入口明确禁用cuDNN和matmul TF32，并记录该设置。严格确定性并不保证不同批形状或不同设备浮点逐值相同，这两个病例的结果不能推广为所有硬件均位级一致。

历史训练/评估源码保持原设置；本轮OAR-Spine效果由冻结评估代码产生，不能用新的精度设置替换历史分数。该数值修正也不能解释或抵消OAR-Spine主指标未提升的结果。

可复查证据：

- `outputs/spine_design_20261027/provenance/input_equivalence.json`：全量输入等价。
- `geometry_tests.log`、`inference_tests_final.log`：3项几何与5项推理测试，含48种变换。
- `real_inference.log`、`real_inference_v2.log`：保留的初次失败。
- `batch_numeric_diagnostic_v2.json`：GPU/CPU、批大小与TF32对比。
- `real_inference_v3/verification.json`：严格FP32最终真实影像检查。

原始来源数据、冻结运行和既有checkpoint均未被修改。性能有效性仍以独立实验为准。
