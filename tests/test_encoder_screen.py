"""Fresh-process frozen execution, report coverage and predeclared decisions."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml

from experiments.run_encoder_text import ARMS, prepare, decision
from test_freqssl import CPUInvariantTest, small_config, make_spider_fixture
from test_hierarchical_text import write_bank


class EncoderScreenTests(CPUInvariantTest):
    def test_decisions_require_controls_union_and_flag_rank_reversal(self):
        arms = {a:dict(best=dict(dice=.8,dice_union_present=.78),
                       last=dict(dice=.8,dice_union_present=.78)) for a in ARMS}
        arms["T2"] = dict(best=dict(dice=.81,dice_union_present=.79),
                          last=dict(dice=.79,dice_union_present=.78))
        result = decision(arms)
        self.assertTrue(result["fusion"]["T2"]["passed"])
        self.assertTrue(result["fusion"]["T2"]["unstable"])
        self.assertIsNone(result["placement"]["T"]["preferred"])
        self.assertIsNone(result["granularity"]["E2"]["preferred"])
        arms["A1"]["best"]["dice_union_present"] = .80
        self.assertFalse(decision(arms)["fusion"]["T2"]["passed"])
        arms["A1"]["best"]["dice_union_present"] = .78
        arms["A1"]["best"]["dice"] = .8095
        self.assertFalse(decision(arms)["fusion"]["T2"]["passed"])

    def _fresh_runner(self, device):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            banks = root / "banks"
            banks.mkdir()
            write_bank(banks)
            data = make_spider_fixture(root/"data")
            cfg = small_config(data,method="mean_teacher",slice_context="adaptive",
                labelnum=4,max_samples=8,max_steps=3,eval_every=3,text_adapter_dim=8)
            path = root/"base.yaml"
            path.write_text(yaml.safe_dump(cfg.to_dict()))
            output = root/"screen"
            prepare(path,banks,output)
            (banks/"text.pt").unlink()
            run = subprocess.run([sys.executable,str(output/"code/experiments/run_encoder_text.py"),
                "--output",str(output),"--device",device,"--execute-frozen"],
                capture_output=True,text=True,timeout=150)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            result = json.loads((output/"results.json").read_text())
            self.assertEqual(result["status"],"completed")
            self.assertEqual(set(result["arms"]),set(ARMS))
            self.assertFalse(result["test_evaluated"])
            self.assertFalse(result["decision"]["confirmatory_efficacy_claim"])
            self.assertEqual(json.loads((output/"exit.json").read_text())["exit_code"],0)
            with (output/"metrics_report/per_class.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows),8*2*20)
            self.assertEqual({r["checkpoint"] for r in rows},{"best","last"})
            with (output/"metrics_report/training_diagnostics.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            for arm in ARMS[2:]:
                self.assertTrue(any(r["arm"]==arm and r[f"E{arm[1]}_residual_rms_ratio"] for r in rows))
            self.assertTrue((output/"metrics_report/paired_case_deltas.csv").is_file())

    def test_frozen_runner_cpu(self):
        self._fresh_runner("cpu")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
    def test_frozen_runner_fresh_cuda_process(self):
        self._fresh_runner("cuda:0")


if __name__ == "__main__":
    unittest.main()
