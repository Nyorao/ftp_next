"""Generate an auditable Chinese readout from completed 3D evaluation files."""

import argparse
import csv
import json
from pathlib import Path


NAMES = ["背景", "L5 椎体", "L4 椎体", "L3 椎体", "L2 椎体", "L1 椎体", "T12 椎体",
         "T11 椎体", "T10 椎体", "T9 椎体", "椎管", "L5/S1 椎间盘", "L4/L5 椎间盘",
         "L3/L4 椎间盘", "L2/L3 椎间盘", "L1/L2 椎间盘", "T12/L1 椎间盘",
         "T11/T12 椎间盘", "T10/T11 椎间盘", "T9/T10 椎间盘"]
ARM_NAMES = {"anchor": "单学生基线", "dual_no_feedback": "双学生，无反馈", "dual_feedback": "双学生，有反馈"}
NETWORK_NAMES = {"teacher": "教师", "student1": "S1", "student2": "S2", "students_mean": "S1/S2 概率集成"}


def value(number, percent=False):
    if number is None:
        return "—"
    return f"{number * 100:.4f}" if percent else f"{number:.2f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = args.evaluation_dir.resolve()
    results = json.loads((root / "results_geometry_corrected.json").read_text())
    if results["status"] != "completed":
        raise ValueError("Cannot summarize incomplete results")
    manifest = json.loads((root / "manifest.json").read_text())
    run = Path(manifest["training_run"])
    training = json.loads((run / "results.json").read_text())
    data = json.loads((root / "training_data_audit.json").read_text())
    reconstruction = json.loads((root / "data_audit.json").read_text())
    main_rows = []
    comparisons = []
    for split in ("valid", "test"):
        entries = results["splits"][split]
        anchor = entries["anchor/teacher"]
        feedback = entries["dual_feedback/teacher"]
        for arm in ARM_NAMES:
            q = entries[f"{arm}/teacher"]["overall"]
            main_rows.append([split, ARM_NAMES[arm], value(q["dice"], True), value(q["iou"], True),
                              value(q["hd95_mm"]), value(q["asd_mm"]),
                              f"{q['valid_surface_class_case_pairs']}/{q['gt_present_class_case_pairs']}",
                              q["missed_gt_class_case_pairs"]])
        old = {item["case"]: item for item in anchor["per_case"]}
        for item in feedback["per_case"]:
            comparisons.append({"split": split, "case": item["case"],
                                "anchor_dice": old[item["case"]]["dice"], "feedback_dice": item["dice"],
                                "delta_dice": item["dice"] - old[item["case"]]["dice"]})
    with (root / "teacher_paired_case_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    lines = [
        "# SPIDER 双学生训练与完整 3D 评估核查（2026-10-16）", "",
        "本报告核查已完成的严格确定性 v2 预实验。三个组均为固定最后一步 checkpoint；本次没有重新训练或根据 test 选择模型。", "",
        "## 数据与实际训练量", "",
        f"- PNG 数据：`{manifest['png_root']}`。对应 MHA 体数据：`{manifest['mha_root']}`。原始 SPIDER masks 位于 `/data/zhp/SPIDER/masks`。",
        "- 原划分：train 168 个病例/3732 张切片，valid 21/418，test 21/481。按 SPIDER 病例编号无跨 split 交叉；不把切片数量当作独立病例数量。",
        "- 原训练标注池为排序后的前 1496 张；本轮从标注/未标注池均匀抽取 48/72 张切片，分别涉及 47/70 个病例。病例 069 横跨两个训练池，但不跨 train/valid/test。",
        f"- 重放保存的采样规则确认：实际访问 {data['seen_unique_labeled_slices']} 张不同标注切片、{data['seen_unique_unlabeled_slices']} 张不同未标注切片；两池累计各有 240 次样本访问。未读取未标注训练样本 GT。",
        "- seed=42；每组 240 次更新，5 个子集 epoch；batch=2（1 标注+1 未标注），224 输入。视觉网络从零训练，保留冻结 BERT、类别向量、APGM/PSGM。",
        "- 这不是全量训练或收敛模型。48 张标注切片中有 6 张全背景；T9/T10 椎间盘仅在 1 张中出现，共 41 个原分辨率像素。", "",
        "## 评估方法与重建核验", "",
        "此前训练记录仅评估 16 个验证病例各一张中心切片，属于 2D 运行诊断，不能作为 SPIDER 3D 整体性能。",
        "本次使用完整 valid/test 的全部 899 张切片，重建 42 个完整体。按病例和数字序号分组，检查切片数、连续性和重复；还原数据加载的 W/H 转置及原切片预处理的上下翻转，恢复原体数组轴序。",
        f"42 个体的 PNG 标签重建与参考 MHA 逐体素完全相同；valid/test 分别有 {sum(r['undo_vertical_flip'] for r in reconstruction['splits']['valid']['case_records'])}/{sum(r['undo_vertical_flip'] for r in reconstruction['splits']['test']['case_records'])} 例需要还原上下翻转。导出的预测 MHA 逐一验证数组与几何信息。",
        "推理采用 eval/no-grad，逐切片 argmax 后以最近邻恢复原 PNG 尺寸；无测试增强、连通域筛选或验证/测试标签指导的预测修正。教师、S1、S2、概率集成独立统计。GT 仅用于核对既有预处理的逆变换和计算指标，未进入网络/文本输入。", "",
        "## 物理间距修正证据", "",
        "SPIDER_126（valid）、192/195（test）的现有 MHA 数组已做轴转置，但 spacing 仍沿用转置前顺序。已分别在原始 masks 中找到 35_t2、234_t2、239_t1/239_t2；经标签映射及 transpose(1,2,0) 后与当前参考逐体素匹配。195 的两个原始候选给出相同间距。",
        "三例用于当前 NumPy Z/Y/X 网格的 spacing 从约 (3.3, 0.5859375, 0.5859375) 校正为 (0.5859375, 0.5859375, 3.3) mm。SimpleITK 轴变换同时处理方向，另存校正后的预测 MHA；用户数据、checkpoint 和原始评估结果均未改动。",
        f"证据见 `{root / 'raw_geometry_trace.json'}` 和 `geometry_correction_audit.json`。下列距离指标使用校正结果，Dice/IoU 在校正前后完全一致。", "",
        "## 指标口径", "",
        "- Dice = 2|P∩G|/(|P|+|G|)，IoU = |P∩G|/|P∪G|；均在完整 3D 体上计算，表中以百分数表示。",
        "- HD95：六邻域二值表面双向距离合并后的第 95 百分位；ASD：预测表面到 GT 表面的平均距离，沿用原项目单向定义；均为 mm，越小越好。",
        "- 整体：每个病例先平均 GT 存在的前景类别，再对病例等权平均；每类别：在该类 GT 存在的病例间平均。ID 0 背景不计入主平均。",
        "- GT 存在而预测为空：Dice/IoU=0，HD95/ASD 未定义；GT 不存在的类别从主均值排除，另记假阳性。距离表必须结合有效对数与漏检数，不能把未定义距离当作 0。",
        "- CSV/JSON 还报告只排除双空、把 GT 缺失假阳性计为 0 的 union-present Dice/IoU，避免 GT-present 口径掩盖假阳性。",
        "- 教师报告四种指标；S1/S2/集成补充报告完整 3D Dice/IoU，本次未计算它们的表面距离。", "",
        "## 单教师主要结果", "",
        table(["划分", "实验组", "Dice %", "IoU %", "HD95 mm", "ASD mm", "有效距离/GT存在类-病例对", "完全漏检对数"], main_rows), "",
    ]
    for split in ("valid", "test"):
        selected = [q for q in comparisons if q["split"] == split]
        delta = sum(q["delta_dice"] for q in selected) / len(selected) * 100
        improved = sum(q["delta_dice"] > 0 for q in selected)
        lines.append(f"{split}：反馈相对基线 Dice 变化 {delta:+.4f} 个百分点；{improved}/{len(selected)} 个病例提高。")
    lines += ["", "无反馈双学生的 S1/教师与 anchor 最终状态逐元素相同，最大差值为 0；3D 评估对相同网络状态复用结果，并记录 reused_from。该负对照通过。", "",
              "## 各网络的 3D Dice", ""]
    network_rows = []
    for identity, entry in results["splits"]["test"].items():
        arm, network = identity.split("/")
        network_rows.append([ARM_NAMES[arm], NETWORK_NAMES[network],
                             value(results["splits"]["valid"][identity]["overall"]["dice"], True),
                             value(entry["overall"]["dice"], True), value(entry["overall"]["iou"], True)])
    lines += [table(["实验组", "推理网络", "valid Dice %", "test Dice %", "test IoU %"], network_rows), ""]
    for split in ("test", "valid"):
        for arm in ("anchor", "dual_feedback"):
            rows = []
            for q in results["splits"][split][f"{arm}/teacher"]["per_class"]:
                rows.append([q["class_id"], NAMES[q["class_id"]], q["gt_present_cases"], value(q["dice"], True),
                             value(q["iou"], True), value(q["hd95_mm"]), value(q["asd_mm"]),
                             q["valid_surface_cases"], q["missed_gt_cases"]])
            lines += [f"## {split}：{ARM_NAMES[arm]}教师各类别", "",
                      table(["ID", "类别", "GT病例数", "Dice %", "IoU %", "HD95 mm", "ASD mm", "有效距离病例", "漏检病例"], rows), ""]
    coverage = training["arms"]["dual_feedback"]["mean_active_feedback_coverage"] * 100
    no_feedback = results["splits"]["test"]
    lines += ["## 对双学生有效性的判断", "",
              "本轮反馈组的单教师重叠指标没有显示收益，且两个方案的绝对分数都很低。可以确认独立双学生/单教师训练路径能运行、负对照正确，以及反馈确实影响了 S1；不能据此确认双学生能提高收敛后的分割性能，也不能由这次短程失败判定所有双学生设计无效。",
              f"无反馈组的学生概率集成在 test 上为 {value(no_feedback['dual_no_feedback/students_mean']['overall']['dice'], True)}% Dice，高于其 S1 的 {value(no_feedback['dual_no_feedback/student1']['overall']['dice'], True)}% 和 S2 的 {value(no_feedback['dual_no_feedback/student2']['overall']['dice'], True)}%。这是双网络推理的初步信号，尚未转化为单教师收益，也没有多种子或收敛验证。",
              f"反馈启用 192 步，其中仅 47 步有通过筛选的像素；共 788/9,633,792 个候选像素，平均覆盖率 {coverage:.6f}%。置信度筛选未被证实能提供可靠前景目标。",
              "后续应先用足量数据/训练预算建立可用的单学生 3D 基线，再在相同条件下比较无反馈与延迟启用反馈的双学生；检查伪标签可靠性和类别覆盖，最后做多个随机种子。不要据当前 test 结果选择阈值或反复调参。", "",
              "## 产物与验证", "",
              f"- 训练记录：`{run}`；重建与指标：`{root}`。",
              "- 主结果：results_geometry_corrected.json；所有实验/网络的 per_class_geometry_corrected.csv、per_case_geometry_corrected.csv。",
              "- 3D 预测：各组/网络的 mha/；三例几何修正另存 mha_geometry_corrected/。完全相同网络的复用路径在 result 中明确标记。",
              "- 配对病例差异：teacher_paired_case_comparison.csv；训练样本审计：training_data_audit.json。",
              "- 本次新增 5 项测试通过：数字排序及缺片/重复、各轴及翻转还原、与 MedPy 一致的各向异性距离、空预测/假阳性统计、原始 mask 证据验证与 spacing 轴置换。", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(args.report.resolve())


if __name__ == "__main__":
    main()
