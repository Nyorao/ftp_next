"""Fresh frozen processes cover all arms, reporting and ten inference interventions."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml

from experiments.run_dual_text import ARMS, prepare, decision
from test_freqssl import CPUInvariantTest, small_config, make_spider_fixture
from test_hierarchical_text import write_bank


class DualScreenTests(CPUInvariantTest):
    def test_predeclared_controls_and_rank_reversals(self):
        arms = {a:dict(best=dict(dice=.8,dice_union_present=.78),
            last=dict(dice=.8,dice_union_present=.78)) for a in ARMS}
        arms["N7"] = dict(best=dict(dice=.81,dice_union_present=.79),last=dict(dice=.79,dice_union_present=.78))
        result = decision(arms)
        self.assertTrue(result["full_module"]["passed"])
        self.assertTrue(result["full_module"]["unstable"])
        self.assertEqual({r["control"] for r in result["full_module"]["comparisons"]},{"N0","N1","N2","N5","N6"})
        for control in ("N0","N1","N2","N5","N6"):
            arms[control]["best"]["dice_union_present"] = .80
            self.assertFalse(decision(arms)["full_module"]["passed"])
            arms[control]["best"]["dice_union_present"] = .78
        self.assertEqual(len(result["component_effects"]["routing"]),2)
        arms["N6"]["best"]["dice"] = .8095
        self.assertFalse(decision(arms)["full_module"]["passed"])

    def _fresh(self,device):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            banks = root/"banks"
            banks.mkdir()
            write_bank(banks)
            data = make_spider_fixture(root/"data")
            cfg = small_config(data,method="mean_teacher",slice_context="adaptive",labelnum=4,
                max_samples=8,max_steps=3,eval_every=3,text_adapter_dim=8)
            path = root/"base.yaml"
            path.write_text(yaml.safe_dump(cfg.to_dict()))
            output = root/"screen"
            prepare(path,banks,output)
            (banks/"text.pt").unlink()
            args = [sys.executable,str(output/"code/experiments/run_dual_text.py"),"--output",str(output),
                "--device",device,"--execute-frozen"]
            process = subprocess.run(args,capture_output=True,text=True,timeout=240)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            result = json.loads((output/"results.json").read_text())
            self.assertEqual(result["status"],"completed")
            self.assertEqual(set(result["arms"]),set(ARMS))
            self.assertEqual(result["post_training_interventions"]["evaluations"],10)
            self.assertTrue(result["post_training_interventions"]["full_reproduced"])
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(json.loads((output/"exit.json").read_text())["exit_code"],0)
            def rows(relative):
                with (output/relative).open(encoding="utf-8-sig") as handle:
                    return list(csv.DictReader(handle))
            self.assertEqual(len(rows("metrics_report/per_class.csv")),8*2*20)
            self.assertEqual(len(rows("interventions/per_class.csv")),2*5*20)
            self.assertEqual(len(rows("interventions/overall.csv")),10)
            samples = rows("metrics_report/training_diagnostics.csv")
            for stage in (2,3):
                self.assertTrue(any(r["arm"]=="N7" and r[f"E{stage}_raw_difference_rms_ratio"] for r in samples))
            windows = rows("metrics_report/training_window_summary.csv")
            self.assertTrue(all(r["samples"]=="3" and r["scope"]=="every_training_step" for r in windows))
            # Explicit runner resume is idempotent: no extra updates/evaluations.
            saved = {p:p.read_bytes() for p in output.glob("runs/*/metrics.jsonl")}
            saved.update({p:p.read_bytes() for p in output.glob("interventions/*_*.json")})
            process = subprocess.run(args,capture_output=True,text=True,timeout=120)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            self.assertTrue(all(p.read_bytes()==value for p,value in saved.items()))

    def test_frozen_runner_cpu(self):
        self._fresh("cpu")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
    def test_frozen_runner_fresh_cuda_process(self):
        self._fresh("cuda:0")


if __name__ == "__main__":
    unittest.main()
