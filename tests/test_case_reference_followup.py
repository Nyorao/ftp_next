import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from freqssl.case_context import CaseFusion
from freqssl.engine import write_json
from freqssl.models import StudentTeacherSystem
from freqssl.evaluation import prepare_validation,evaluate_volumes
from experiments.probe_case_reference import FusionProbe,controls,probe
from experiments.run_case_reference_followup import decomposition,summarize
from test_case_context import cfg_for,fake_geometry
from test_freqssl import CPUInvariantTest,make_spider_fixture


class FollowupTests(CPUInvariantTest):
    def test_four_slot_gamma_algebra_and_padded_probe(self):
        model=StudentTeacherSystem(cfg_for('B')).students[0].eval()
        f=model.context_fusion[0];f.gamma.data.fill_(.6)
        x=torch.randn(8,4,model.widths[2],8,8)
        torch.testing.assert_close(f(x),.55*x[:,1]+.15*(x[:,0]+x[:,2]+x[:,3]))
        capture=FusionProbe(model);capture.valid=3
        with torch.no_grad():
            for layer,channels,size in zip(model.context_fusion,model.widths[2:],(8,4,2)):
                layer(torch.randn(8,4,channels,size,size))
        result=capture.finish_case();capture.close()
        self.assertEqual(result[0]['real_slices'],3)
        self.assertAlmostEqual(result[0]['weight'][3]['mean'],.25)
        self.assertAlmostEqual(result[0]['effective_coefficient'][3]['mean'],.15,places=6)
        self.assertAlmostEqual(result[0]['effective_coefficient'][1]['mean'],.55,places=6)

    def test_cross_case_maps_preserve_marginal_and_never_use_same_case(self):
        groups={'a':list(range(2)),'b':list(range(3)),'c':list(range(4))}
        positions={c:np.linspace(-1,1,len(v)).tolist() for c,v in groups.items()}
        for seed in (1701,1702,1703):
            result=controls(groups,positions,seed)
            self.assertEqual(result,controls(groups,positions,seed))
            self.assertTrue(all(c!=d for c,d in result['reference_donor'].items()))
            src=[];dst=[]
            for c,mapping in result['position_mapping'].items():
                for i,row in mapping.items():
                    self.assertNotEqual(c,row['donor_case']);dst.append(row['position']);src.append(positions[c][i])
                    self.assertNotEqual(result['random_indices'][c][i],(len(groups[c])-1)//2)
            self.assertCountEqual(src,dst)

    def run_probe(self,device):
        with tempfile.TemporaryDirectory() as tmp,mock.patch('freqssl.case_context.verify_dataset_geometry',side_effect=fake_geometry):
            root=Path(tmp);data=make_spider_fixture(root/'data')
            for folder in ('image','label'):
                for f in (data/'valid'/folder).glob('*.png'):
                    shutil.copy2(f,f.with_name(f.name.replace('003','005')))
                mha=data.parent/'mha/valid'/folder
                shutil.copy2(mha/'SPIDER_003.mha',mha/'SPIDER_005.mha')
            for arm in 'BC':
                cfg=cfg_for(arm,root=data,labelnum=4,max_samples=8)
                system=StudentTeacherSystem(cfg).to(device)
                for layer in system.teachers[0].context_fusion: layer.gamma.data.fill_(.5)
                path=root/f'{arm}.pt'
                torch.save(dict(format_version=3,config=cfg.to_dict(),model=system.state_dict(),step=6000),path)
                ds,records=prepare_validation(cfg)
                expected=evaluate_volumes(system,ds,records,device)['teacher1']['overall']
                actual=probe(path,root/f'{arm}_analysis',device,expected_cases=2)
                self.assertEqual(actual['metrics']['full']['overall'],expected)
                self.assertEqual(len(actual['conditions']),8 if arm=='B' else 4)
                self.assertEqual(actual['fusion'][0]['mean_w_case_equal'],[.25]*4 if arm=='B' else [1/3]*3) if arm=='B' else None
                self.assertTrue(actual['model_state_unchanged'])
                self.assertEqual(actual['per_case_fusion']['SPIDER_003'][0]['real_slices'],2)

    def test_probe_cpu_native_scores_and_all_controls(self): self.run_probe('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_probe_cuda_native_scores_and_all_controls(self): self.run_probe('cuda:0')

    def test_fixed_step_summary_and_union_decomposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);baseline=root/'baseline';out=root/'out';out.mkdir()
            write_json(out/'manifest.json',dict(baseline=str(baseline)))
            for seed in (42,43,44):
                for arm in ('A','B','C') if seed==42 else ('A','B'):
                    directory=baseline/f'screen/{arm}' if seed==42 else out/f'training/seed{seed}/{arm}'
                    directory.mkdir(parents=True)
                    write_json(directory/'result.json',dict(status='completed',step=6000))
                    cases=[];d=.8+.001*(seed-42)+(.004 if arm=='B' else 0)
                    for i in range(21):
                        rows=[dict(class_id=j,class_name=str(j),gt_voxels=100 if j==1 else 0,
                            pred_voxels=100 if j==1 else (1 if j==2 and arm=='B' else 0),
                            dice=d if j==1 else None,dice_union_present=d if j==1 else (0 if j==2 and arm=='B' else None),
                            false_positive_absent_class=j==2 and arm=='B') for j in range(1,20)]
                        cases.append(dict(case=f'case{i}',classes=rows))
                    val=dict(cases=cases,overall=dict(cases=21,dice=d,dice_union_present=d/(2 if arm=='B' else 1)))
                    write_json(directory/'valid_step_6000.json',dict(teacher1=val))
                    # Deliberately unusable best metadata: final reporting must never read it.
                    (directory/'best.pt').write_text('not the final teacher')
            r=summarize(out)
            self.assertEqual(r['status'],'completed')
            self.assertAlmostEqual(r['mean_std']['A']['dice']['sample_std'],.1)
            self.assertEqual(r['case_bootstrap_after_seed_average']['primary_dice_percent']['n_cases'],21)
            self.assertAlmostEqual(r['case_bootstrap_after_seed_average']['primary_dice_percent']['mean'],.4)
            self.assertTrue((out/'per_class_mean_std.csv').exists())

if __name__=='__main__': unittest.main()
