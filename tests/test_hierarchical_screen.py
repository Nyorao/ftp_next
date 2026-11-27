"""Exercise the frozen multi-arm runner through native-volume report export."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
import yaml

from experiments.run_hierarchical_text import ARMS, prepare, decision
from test_freqssl import CPUInvariantTest, small_config, make_spider_fixture
from test_hierarchical_text import write_bank


class HierarchicalScreenTests(CPUInvariantTest):
    def test_screen_requires_all_controls_and_preserves_union_metric(self):
        protocol=dict(threshold_vs_context_pp=.3,threshold_vs_neutral_pp=.1,threshold_vs_last_layer_pp=.1)
        arms={a:dict(overall=dict(dice=.8,dice_union_present=.78)) for a in ARMS}
        arms['hierarchical']['overall']=dict(dice=.81,dice_union_present=.79)
        self.assertTrue(decision(arms,protocol)['passed_single_seed_screen'])
        arms['last_layer']['overall']['dice']=.8095
        self.assertFalse(decision(arms,protocol)['passed_single_seed_screen'])
        arms['last_layer']['overall']['dice']=.8
        arms['neutral']['overall']['dice_union_present']=.80
        self.assertFalse(decision(arms,protocol)['passed_single_seed_screen'])

    def _frozen_execution_and_report(self, device):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            banks=root/'banks'
            banks.mkdir()
            write_bank(banks,'text')
            write_bank(banks,'neutral')
            data=make_spider_fixture(root/'data')
            cfg=small_config(data,method='mean_teacher',slice_context='adaptive',
                             labelnum=4,max_samples=8,max_steps=3,eval_every=3)
            config=root/'base.yaml'
            config.write_text(yaml.safe_dump(cfg.to_dict()))
            output=root/'screen'
            prepare(config,banks,output)
            # Reproducibility must not depend on the mutable original banks.
            for p in banks.iterdir():
                p.unlink()
            command=[sys.executable,str(output/'code/experiments/run_hierarchical_text.py'),
                     '--output',str(output),'--device',device,'--execute-frozen']
            run=subprocess.run(command,capture_output=True,text=True,timeout=90)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            results=json.loads((output/'results.json').read_text())
            self.assertEqual(results['status'],'completed')
            self.assertEqual(set(results['arms']),set(ARMS))
            self.assertFalse(results['test_evaluated'])
            self.assertFalse(results['decision']['confirmatory_efficacy_claim'])
            self.assertEqual(json.loads((output/'exit.json').read_text())['exit_code'],0)
            with (output/'metrics_report/per_class.csv').open(encoding='utf-8-sig') as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),80)
            self.assertEqual({int(r['class_id']) for r in rows},set(range(20)))
            self.assertTrue((output/'metrics_report/report.md').exists())
            with (output/'metrics_report/overall.csv').open(encoding='utf-8-sig') as handle:
                rows=list(csv.DictReader(handle))
            self.assertTrue(all(r['final_step_iou_pct'] and r['final_step_union_dice_pct'] for r in rows))
            with (output/'metrics_report/training_diagnostics.csv').open(encoding='utf-8-sig') as handle:
                rows=list(csv.DictReader(handle))
            text_rows=[r for r in rows if r['arm']!='context']
            self.assertTrue(text_rows)
            self.assertTrue(all(r['E4_layer_weight_1'] and r['D2_tissue_fraction_0'] for r in text_rows))
            self.assertTrue((output/'metrics_report/paired_case_deltas.csv').exists())
            # Re-execution reuses completed arms and verifies the frozen hashes.
            (output/'protocol.json').write_text('{}')
            run=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('Frozen protocol/source/config/bank changed',run.stderr)

    def test_four_arm_frozen_execution_and_report(self):
        self._frozen_execution_and_report('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_fresh_cuda_process_runs_frozen_screen(self):
        # The subprocess has no context allocated by earlier tests. This catches
        # the allocator reset-before-init bug that ordinary CUDA unit tests miss.
        self._frozen_execution_and_report('cuda:0')


if __name__=='__main__':
    unittest.main()
