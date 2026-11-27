"""Frozen E2-token/E3-CLS factorial screen and fixed-checkpoint interventions."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments import run_encoder_text as shared
from experiments.run_hierarchical_text import initialize_device, class_name
from experiments.run_slice_context import _SelectedTeacher, selected_validation
from freqssl.config import load_config
from freqssl.dual_text import INTERVENTIONS
from freqssl.engine import run_experiment, seed_everything, state_hash, write_json
from freqssl.evaluation import prepare_validation, evaluate_volumes
from freqssl.inference import load_single_network
from freqssl.prompt_supervision import file_sha256
from spider_volume import aggregate_cases

ARMS = tuple(f"N{i}" for i in range(8))
# All nonbaseline arms instantiate the same tensors before disabling unused paths.
SETTINGS = {
    "N1": ((), "independent", "plain"),
    "N2": ((2, 3), "independent", "plain"),
    "N3": ((2, 3), "aligned", "plain"),
    "N4": ((2, 3), "independent", "difference"),
    "N5": ((2,), "aligned", "difference"),
    "N6": ((3,), "aligned", "difference"),
    "N7": ((2, 3), "aligned", "difference"),
}
LABELS = dict(N0="视觉基线", N1="仅共同组织监督", N2="双分支：独立路由＋直接残差",
    N3="双分支：组织路由＋直接残差", N4="双分支：独立路由＋差分残差",
    N5="仅 E2 Token：组织路由＋差分", N6="仅 E3 CLS：组织路由＋差分",
    N7="E2 Token＋E3 CLS：组织路由＋差分")
read = shared.read


def arm_config(base, arm, bank, output):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    stages, routing, residual = SETTINGS.get(arm, ((), "independent", "plain"))
    return replace(base, output_dir=str(output), internal_text="none" if arm == "N0" else "text",
        text_bank="" if arm == "N0" else str(bank), text_fusion_layout="e4" if arm == "N0" else "encoder_multi",
        text_layer_mode="last", text_feature_kind="none", encoder_text_stage=None,
        encoder_text_stages=stages, text_routing_mode=routing, text_residual_mode=residual,
        text_alignment_weight=0. if arm == "N0" else .1, text_alignment_balance="class").validate()


def prepare(config, banks, output):
    shared.prepare(config, banks, output, arms=ARMS, make_config=arm_config, protocol_extra=dict(
        screen="E2 word tokens plus E3 per-class CLS; routing x residual factorial",
        arm_settings={arm:dict(stages=list(s), routing=r, residual=d) for arm,(s,r,d) in SETTINGS.items()},
        placement="E2 token and E3 CLS after block, before skip storage/downsampling; E4 auxiliary only",
        representation="fixed layer12; E2 valid word tokens; E3 one CLS per tissue; 768D",
        routing="aligned: same pre-injection logits for auxiliary CE and class mixing; no detach",
        difference="Phi(Q,Z)-Phi(Q,0); shared parameters; both branches on graph",
        fusion_threshold_vs_A0_pp=None, fusion_threshold_vs_A1_pp=None, preference_threshold_pp=None,
        full_module_thresholds_pp=dict(N0=.3, N1=.1, N2=.1, N5=.1, N6=.1),
        component_contrasts=dict(routing=[["N7","N4"],["N3","N2"]],
            difference=[["N7","N3"],["N4","N2"]], auxiliary=[["N1","N0"]]),
        component_threshold="report signed effects and union; no extra post-hoc pass threshold",
        post_training=dict(arm="N7", checkpoints=["best","last"], states=list(INTERVENTIONS),
            content_permutation=[3,0,1,2], anchors_unchanged=True, full_reproduction_required=True,
            checkpoint_reselection=False, retraining=False),
        diagnostics="sampled first/log_every step; all-step scalar means exported separately"))


def decision(arms):
    rows = [shared.compare(arms, "N7", control, threshold)
            for control,threshold in (("N0",.3),("N1",.1),("N2",.1),("N5",.1),("N6",.1))]
    effects = {}
    for component,pairs in dict(routing=[("N7","N4"),("N3","N2")],
            difference=[("N7","N3"),("N4","N2")], auxiliary=[("N1","N0")]).items():
        values = []
        for candidate,control in pairs:
            row = shared.compare(arms,candidate,control,0.)
            row.pop("passed")
            row.pop("threshold_pp")
            row["positive_best_effect_with_union"] = row["delta_dice_pp"] > 0 and row["delta_union_pp"] >= 0
            values.append(row)
        effects[component] = values
    return dict(full_module=dict(comparisons=rows, passed=all(r["passed"] for r in rows),
        unstable=any(r["rank_reversal"] for r in rows)), component_effects=effects,
        confirmatory_efficacy_claim=False, semantic_understanding_proven=False)


def render_decision(result):
    full = result["full_module"]
    lines = ["", "## 预声明判定", "", f"N7筛查通过={full['passed']}；最佳/末步排序反转={full['unstable']}。"]
    for row in full["comparisons"]:
        lines.append(f"- N7−{row['control']}：最佳Dice {row['delta_dice_pp']:+.4f}pp，"
            f"末步Dice {row['final_delta_dice_pp']:+.4f}pp，最佳Union {row['delta_union_pp']:+.4f}pp；通过={row['passed']}。")
    for component,rows in result["component_effects"].items():
        for row in rows:
            lines.append(f"- {component}，{row['candidate']}−{row['control']}：最佳Dice {row['delta_dice_pp']:+.4f}pp，"
                f"Union {row['delta_union_pp']:+.4f}pp；排序反转={row['rank_reversal']}。")
    lines.append("组件效果按对应消融分别报告；正向数值不代表统计显著性，不用整体收益替代组件证据。")
    return lines


def summarize(output):
    output = Path(output)
    result = shared.summarize(output, arm_order=ARMS, labels=LABELS, decide=decision,
        title="E2 Token＋E3 CLS：语义路由差分融合", render_decision=render_decision)
    path = output / "interventions/result.json"
    result["post_training_interventions"] = read(path) if path.exists() else dict(status="pending")
    if result["status"] == "completed" and result["post_training_interventions"]["status"] != "completed":
        result["status"] = "training_completed"
    write_json(output / "results.json", result)
    report = output / "metrics_report/report.md"
    with report.open("a") as handle:
        handle.write(f"\n训练后固定检查点评估：{result['post_training_interventions']['status']}。"
                     "详细结果见interventions目录；关闭文本和错配只解释训练后依赖，不能替代从头训练消融。\n")
    return result


def metric_identity(score):
    return {k:score[k] for k in ("overall", "per_class", "per_case", "cases")}


def export_interventions(directory, results):
    tables = {name:[] for name in ("overall", "per_class", "per_case", "merged_tissue", "paired_case_deltas")}
    for checkpoint, variants in results.items():
        reference = {row["case"]:row for row in variants["full"]["metrics"]["per_case"]}
        for state,payload in variants.items():
            score = payload["metrics"]
            if aggregate_cases(score["cases"]) != {k:score[k] for k in ("overall","per_class","per_case")}:
                raise ValueError("Intervention metrics disagree with confusion counts")
            common = dict(checkpoint=checkpoint, state=state, step=payload["step"])
            full = variants["full"]["metrics"]["overall"]
            tables["overall"].append(dict(**common, mean_dice_pct=100*score["overall"]["dice"],
                mean_iou_pct=100*score["overall"]["iou"], union_dice_pct=100*score["overall"]["dice_union_present"],
                delta_dice_vs_full_pp=100*(score["overall"]["dice"]-full["dice"]),
                delta_union_vs_full_pp=100*(score["overall"]["dice_union_present"]-full["dice_union_present"]),
                valid_cases=score["overall"]["cases"], seconds=payload["seconds"]))
            for row in score["per_class"]:
                tables["per_class"].append(dict(**common, class_id=row["class_id"], label=class_name(row["class_id"]),
                    dice_pct=shared.percent(row["dice"]), iou_pct=shared.percent(row["iou"]),
                    union_dice_pct=shared.percent(row["dice_union_present"]), gt_present_cases=row["gt_present_cases"]))
            background = []
            for case in score["cases"]:
                cm = np.asarray(case["anatomy"]["confusion_gt_rows_pred_columns"], dtype=np.int64)
                gt,pred,tp = int(cm[0].sum()),int(cm[:,0].sum()),int(cm[0,0])
                if gt:
                    background.append((2*tp/(gt+pred),tp/(gt+pred-tp)))
            tables["per_class"].append(dict(**common, class_id=0, label="背景",
                dice_pct=100*float(np.mean([v[0] for v in background])) if background else None,
                iou_pct=100*float(np.mean([v[1] for v in background])) if background else None,
                gt_present_cases=len(background)))
            for row in score["per_case"]:
                tables["per_case"].append(dict(**common, case=row["case"], dice_pct=shared.percent(row["dice"]),
                    iou_pct=shared.percent(row["iou"]), union_dice_pct=shared.percent(row["dice_union_present"])))
                other = reference[row["case"]]
                tables["paired_case_deltas"].append(dict(**common, case=row["case"],
                    dice_delta_pp=100*(row["dice"]-other["dice"]), iou_delta_pp=100*(row["iou"]-other["iou"]),
                    union_delta_pp=100*(row["dice_union_present"]-other["dice_union_present"])))
            for tissue in ("vertebra","canal","disc"):
                values = [c["anatomy"]["tissue_dice"][tissue] for c in score["cases"]]
                values = [v for v in values if v is not None]
                tables["merged_tissue"].append(dict(**common, tissue=tissue,
                    dice_pct=100*float(np.mean(values)) if values else None, gt_present_cases=len(values)))
    for name,rows in tables.items():
        shared.write_csv(directory / f"{name}.csv", rows)


def interventions(output, device, *, run_directory=None):
    output, device = Path(output), torch.device(device)
    directory = output / "interventions"
    directory.mkdir(exist_ok=True)
    run = Path(run_directory) if run_directory is not None else output / "runs/N7"
    best_step, best_score = selected_validation(run)
    last_step = read(run / "result.json")["step"]
    expected = dict(best=(best_step,best_score),
        last=(last_step,read(run / f"valid_step_{last_step}.json")["teacher1"]))
    results = {}
    for checkpoint in ("best","last"):
        path = run / f"{checkpoint}.pt"
        digest = file_sha256(path)
        selected,cfg,receipt = load_single_network(path,device,"teacher1")
        if (cfg.encoder_text_stages != (2,3) or cfg.text_routing_mode != "aligned"
                or cfg.text_residual_mode != "difference" or receipt["checkpoint_step"] != expected[checkpoint][0]):
            raise ValueError("Interventions require the selected N7 checkpoint")
        model = _SelectedTeacher(selected).eval()
        seed_everything(cfg)
        dataset, records = prepare_validation(cfg)
        before = state_hash(selected)
        results[checkpoint] = {}
        for state in INTERVENTIONS:
            write_json(output / "status.json", dict(status="running", phase="fixed_checkpoint_interventions",
                arm="N7", checkpoint=checkpoint, state=state, pid=os.getpid(), device=str(device)))
            selected.backbone.encoder_text.set_intervention(state)
            saved = directory / f"{checkpoint}_{state}.json"
            identity = dict(checkpoint_sha256=digest, checkpoint=checkpoint, state=state,
                step=receipt["checkpoint_step"], frozen_sources_sha256=file_sha256(output/"frozen_sources.json"),
                permutation=[3,0,1,2] if state=="class_mismatch" else None)
            if saved.exists():
                payload = read(saved)
                if any(payload.get(k) != v for k,v in identity.items()):
                    raise ValueError("Saved intervention belongs to another checkpoint/protocol")
            else:
                started = time.monotonic()
                score = evaluate_volumes(model,dataset,records,device,networks=("teacher1",),
                    batch_size=cfg.eval_batch_size)["teacher1"]
                payload = dict(**identity, metrics=score, seconds=time.monotonic()-started)
                write_json(saved,payload)
            if state == "full" and metric_identity(payload["metrics"]) != metric_identity(expected[checkpoint][1]):
                raise AssertionError("Full inference did not reproduce saved validation; stop before interventions")
            results[checkpoint][state] = payload
            print(f"INTERVENTION {checkpoint}/{state}: Dice={100*payload['metrics']['overall']['dice']:.4f}%",flush=True)
        selected.backbone.encoder_text.set_intervention("full")
        if state_hash(selected) != before:
            raise AssertionError("Interventions changed model weights or frozen text/anchors")
        del selected,model,dataset,records
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    export_interventions(directory,results)
    write_json(directory / "result.json", dict(status="completed", checkpoints=["best","last"],
        states=list(INTERVENTIONS), evaluations=10, full_reproduced=True, weights_and_anchors_unchanged=True,
        checkpoint_reselection=False, training_updates=0, test_evaluated=False, semantic_understanding_proven=False))
    (directory / "report.md").write_text("# 固定检查点依赖诊断\n\n最佳及末步完整路径均复现历史验证。"
        "五种状态均覆盖完整验证集，错配固定为[3,0,1,2]，组织锚保持不变。\n\n"
        "overall.csv中的差值为干预−完整模块；关闭分支只解释训练后依赖，不等同于重新训练消融，"
        "也不能单独证明医学语义理解。逐类、背景、逐病例、合并组织及逐病例差值分别输出。\n")


def execute(output, device):
    output = Path(output)
    if ROOT != output / "code":
        raise RuntimeError("Execute the frozen runner")
    if any(file_sha256(output / p) != h for p,h in read(output/"frozen_sources.json").items()):
        raise ValueError("Frozen source/config/bank/protocol changed")
    started, arm, phase = datetime.now(timezone.utc).isoformat(), None, "training"
    with (output/"screen.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for arm in ARMS:
                cfg = load_config(output/"configs"/f"{arm}.yaml")
                directory = Path(cfg.output_dir)
                if (directory/"result.json").exists() and read(directory/"result.json")["status"] == "completed":
                    continue
                write_json(output/"status.json",dict(status="running",phase=phase,arm=arm,
                    pid=os.getpid(),device=str(device),started_utc=started))
                initialize_device(device)
                checkpoint = directory/"last.pt"
                run_experiment(cfg,device,resume=checkpoint if checkpoint.exists() else None)
                summarize(output)
                gc.collect()
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
            phase = "fixed_checkpoint_interventions"
            initialize_device(device)
            interventions(output,device)
            result = summarize(output)
            write_json(output/"status.json",dict(status="completed",decision=result["decision"],
                post_training_interventions=result["post_training_interventions"]))
            write_json(output/"exit.json",dict(exit_code=0,started_utc=started,
                finished_utc=datetime.now(timezone.utc).isoformat()))
        except BaseException as error:
            write_json(output/"status.json",dict(status="failed",phase=phase,arm=arm,error=repr(error)))
            write_json(output/"exit.json",dict(exit_code=1,error=repr(error),
                finished_utc=datetime.now(timezone.utc).isoformat()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/v3/context_efficiency_reference.yaml")
    parser.add_argument("--banks",type=Path,default=ROOT/"outputs/hierarchical_text_assets_20261108")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:2")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--execute-frozen",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--summarize-only",action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize_only:
        summarize(output)
    elif args.execute_frozen:
        execute(output,args.device)
    else:
        if not args.resume:
            prepare(args.config,args.banks,output)
        if not args.prepare_only:
            subprocess.run([sys.executable,"-u",str(output/"code/experiments/run_dual_text.py"),
                "--output",str(output),"--device",args.device,"--execute-frozen"],check=True,cwd=output/"code")
        else:
            print(f"Prepared: {output}",flush=True)


if __name__ == "__main__":
    main()
