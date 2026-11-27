"""Paired inputs, continuous long schedule, exact resume, frozen GPU runner and reports."""
from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import yaml
from experiments.run_dual_text_long import ARMS, arm_config, prepare, verify_pairing, decisions
from experiments.train_long_v3 import resolve_long_config, run_long
from freqssl.engine import run_experiment
from test_freqssl import CPUInvariantTest, small_config, make_spider_fixture
from test_hierarchical_text import write_bank


class DualLongTests(CPUInvariantTest):
    def fixture(self,root):
        bankdir=root/'banks'; bankdir.mkdir(); write_bank(bankdir)
        data=make_spider_fixture(root/'data')
        base=small_config(data,method='mean_teacher',slice_context='adaptive',labelnum=4,max_samples=8,text_adapter_dim=8)
        return base,bankdir

    def test_controls_pairing_and_epoch_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base,banks=self.fixture(root)
            resolved,spe,_=resolve_long_config(base,root/'long',(100,200),5)
            configs={a:arm_config(resolved,a,banks/'text.pt',root/a) for a in ARMS}
            self.assertEqual(resolved.max_steps,200*spe)
            self.assertEqual(resolved.eval_every,5*spe)
            self.assertEqual(configs['MT'].slice_context,'none')
            self.assertEqual(configs['N0'].internal_text,'none')
            self.assertEqual(configs['N0'].text_alignment_weight,0)
            self.assertEqual(configs['N7'].encoder_text_stages,(2,3))
            self.assertEqual(configs['N7'].text_alignment_weight,.1)
            self.assertEqual(configs['N7'].max_seconds,86400)
            receipt=verify_pairing(configs)
            self.assertEqual(len(set(receipt['hashes'].values())),1)
            arms={a:dict(best=dict(dice=.8,dice_union_present=.78),last=dict(dice=.8,dice_union_present=.78)) for a in ARMS}
            arms['N7']=dict(best=dict(dice=.804,dice_union_present=.78),last=dict(dice=.79,dice_union_present=.78))
            self.assertTrue(decisions(arms)['primary']['passed'])
            self.assertTrue(decisions(arms)['primary']['rank_reversal'])
            arms['N0']['best']['dice_union_present']=.79
            self.assertFalse(decisions(arms)['primary']['passed'])

    def _resume(self,device):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base,banks=self.fixture(root)
            for arm in ARMS:
                cfg=arm_config(base,arm,banks/'text.pt',root/arm)
                staged=root/arm/'staged'
                resolved,spe,_=resolve_long_config(cfg,staged,(1,2),1)
                direct=run_experiment(replace(resolved,output_dir=str(root/arm/'direct')),device)
                # Simulated interruption after the first immutable milestone.
                def interrupt(epoch):
                    if epoch==1: raise RuntimeError('test interruption after milestone')
                with self.assertRaisesRegex(RuntimeError,'test interruption'):
                    run_long(cfg,device,staged,(1,2),1,on_milestone=interrupt)
                first=(staged/'epochs/0001/at_epoch.pt').read_bytes()
                if arm=='N7': (banks/'text.pt').unlink()
                result=run_long(cfg,device,staged,(1,2),1,resume=True)
                self.assertEqual(result['status'],'completed')
                final=json.loads((staged/'training/result.json').read_text())
                for key in ('final_student1_hash','final_teacher1_hash','final_validation'):
                    self.assertEqual(direct[key],final[key])
                self.assertEqual(first,(staged/'epochs/0001/at_epoch.pt').read_bytes())
                logs=lambda p:[json.loads(line) for line in p.read_text().splitlines()]
                actual=logs(staged/'training/metrics.jsonl')
                self.assertEqual(logs(root/arm/'direct/metrics.jsonl'),actual)
                train=[r for r in actual if r['kind']=='train']
                self.assertLess(train[spe]['lr'],train[spe-1]['lr'])
                calls=[json.loads(p.read_text()) for p in (staged/'invocations').glob('*.json')]
                self.assertEqual(len(calls),2)
                self.assertEqual(sorted((r['start_step'],r['completed_step']) for r in calls),[(0,2),(2,4)])
                self.assertTrue(all(r['seconds']>0 for r in calls))

    def test_exact_long_resume_cpu_all_three(self): self._resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_exact_long_resume_cuda_all_three(self): self._resume('cuda:0')

    def _fresh(self,device):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base,banks=self.fixture(root)
            path=root/'base.yaml'; path.write_text(yaml.safe_dump(base.to_dict()))
            output=root/'screen'; prepare(path,banks,output,epochs=(1,2),eval_every_epochs=1)
            (banks/'text.pt').unlink()
            args=[sys.executable,str(output/'code/experiments/run_dual_text_long.py'),'--output',str(output),'--device',device,'--execute-frozen']
            p=subprocess.run(args,capture_output=True,text=True,timeout=240)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            read=lambda p:json.loads(p.read_text())
            result=read(output/'results.json')
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['completed_arms'],list(ARMS))
            self.assertEqual(set(result['decision']),{'1','2'})
            self.assertTrue(result['post_training_interventions']['full_reproduced'])
            self.assertEqual(read(output/'exit.json')['exit_code'],0)
            def rows(name):
                with (output/'metrics_report'/f'{name}.csv').open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))
            self.assertEqual(len(rows('overall')),3*2*2)
            self.assertEqual(len(rows('per_class')),3*2*2*20)
            self.assertEqual(len(rows('validation_history')),3*2)
            self.assertEqual(len(rows('comparisons')),2*3)
            self.assertEqual({r['checkpoint'] for r in rows('overall')},{'best','exact'})
            resources=rows('resources')
            for arm in ARMS:
                first=next(r for r in resources if r['arm']==arm and r['milestone_epoch']=='1')
                last=next(r for r in resources if r['arm']==arm and r['milestone_epoch']=='2')
                self.assertEqual(first['invocations'],'1'); self.assertEqual(last['invocations'],'2')
                self.assertGreater(float(last['seconds']),float(first['seconds']))
            windows=rows('training_window_summary')
            self.assertTrue(all(int(r['samples'])==int(r['milestone_epoch'])*2 for r in windows))
            report=(output/'metrics_report/report.md').read_text()
            self.assertNotIn('6000',report); self.assertNotIn('八组',report)
            saved={p:p.read_bytes() for p in output.glob('runs/*/training/metrics.jsonl')}
            saved.update({p:p.read_bytes() for p in output.glob('runs/*/epochs/*/*.pt')})
            p=subprocess.run(args,capture_output=True,text=True,timeout=120)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            self.assertTrue(all(p.read_bytes()==value for p,value in saved.items()))
            # Tampering must stop before any training and produce failure records.
            with (output/'code/freqssl/dual_text.py').open('a') as f: f.write('\n# deliberate test mutation\n')
            p=subprocess.run(args,capture_output=True,text=True,timeout=120)
            self.assertNotEqual(p.returncode,0)
            self.assertEqual(read(output/'status.json')['status'],'failed')
            self.assertEqual(read(output/'status.json')['phase'],'preflight')
            self.assertEqual(read(output/'exit.json')['exit_code'],1)
            self.assertTrue(all(p.read_bytes()==value for p,value in saved.items()))

    def test_frozen_three_arm_runner_cpu(self): self._fresh('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_frozen_three_arm_runner_fresh_cuda(self): self._fresh('cuda:0')


if __name__=='__main__': unittest.main()
