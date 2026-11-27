"""Revision-specific leakage, geometry, routing and fixed decision invariants."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from torch.nn import functional as F
from freqssl.data import build_training_data
from freqssl.engine import run_experiment, train_step, make_optimizer
from freqssl.models import StudentTeacherSystem
from freqssl.inference import load_single_network
from freqssl.objectives import masked_pseudo_loss
from freqssl.prompt_supervision import prompt_routed_loss,PromptCache,file_sha256
from freqssl.prompt_revision import (prompt_bank,make_partition,read_partition,unpack_template_probabilities,
    competence_eligible,gate_stage,CompetenceGate,qualify_cell,teacher_screen,choose_teacher)
from prompt_teacher.revision import balanced_js
from test_freqssl import CPUInvariantTest,make_spider_fixture,small_config,synthetic_batch
from test_prompt_context import fake_cache


def fake_v2(root,split,size=32):
    fake_cache(root,split,size)
    m=json.loads((root/'manifest.json').read_text());m.update(format_version=2,prompt_bank=prompt_bank('text')[:4],samples_per_template=2,teacher_variant='T1')
    for name,e in m['entries'].items():
        with np.load(root/e['file']) as d:q=d['probabilities']
        np.savez(root/e['file'],template_probabilities=np.stack([q]*4),valid=np.ones((size,size),bool))
        e['sha256']=file_sha256(root/e['file'])
    (root/'manifest.json').write_text(json.dumps(m));return root


def fake_gate(path,enable=True):
    value=dict(format_version=1,status='completed',source='text',variant='T1',prompt_bank=prompt_bank('text')[:4],
        tables={str(s):[[enable]*2 for _ in range(4)] for s in range(500,6000,500)})
    path.write_text(json.dumps(value));return path


class RevisionTests(CPUInvariantTest):
    def test_calibration_pipeline_builds_all_stages_from_case_counts(self):
        from prompt_teacher.calibration import calibrate
        class EMA(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images),20,16,16)
        class Reference(torch.nn.Module):
            def __init__(self):
                super().__init__();self.teachers=torch.nn.ModuleList([EMA()])
        class Dataset:
            def __init__(self, *args, **kwargs):
                self.names=kwargs['names'];self.epochs=[]
            def __len__(self):return len(self.names)
            def set_epoch(self, epoch):self.epochs.append(epoch)
            def __getitem__(self, i):
                targets=torch.ones(4,16,16);targets[0]=2;targets[3]=0
                return dict(image=torch.zeros(3,16,16),label=torch.full((16,16),10),
                    prompt_targets=targets,file_name=self.names[i])
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);cases=[f'SPIDER_{i:03d}' for i in range(1,6)]
            names=[c+'_0_1.png' for c in cases]
            partition=dict(calibration_files=names,calibration_cases=cases)
            split=dict(labeled_files=['SPIDER_100_0_1.png'],unlabeled_files=['SPIDER_101_0_1.png'])
            partition_path=tmp/'partition.json';partition_path.write_text(json.dumps(partition))
            cfg=small_config(tmp,method='mean_teacher',training_split=str(partition_path))
            labels=tmp/'train/label';labels.mkdir(parents=True)
            for name in names:(labels/name).write_bytes(b'authorized fixture')
            reference=tmp/'reference';reference.mkdir();(reference/'split.json').write_text(json.dumps(split))
            for step in range(500,6000,500):(reference/f'step_{step}.pt').write_bytes(b'reference fixture')
            cache=tmp/'cache';cache.mkdir();(cache/'manifest.json').write_text('{}')
            def load(path,device):return Reference(),cfg,{'step':int(path.stem.split('_')[1])}
            with mock.patch('prompt_teacher.calibration.build_training_data',return_value=(None,None,split)), \
                 mock.patch('prompt_teacher.calibration.SpiderSliceDataset',Dataset), \
                 mock.patch('prompt_teacher.calibration.CalibrationCache'), \
                 mock.patch('prompt_teacher.calibration.load_model',side_effect=load), \
                 mock.patch('torch.cuda.max_memory_allocated',return_value=0):
                calibrate(cfg,reference,cache,tmp/'gate','text','T1',torch.device('cpu'))
            result=json.loads((tmp/'gate/gate.json').read_text())
            self.assertEqual(len(result['tables']),11)
            self.assertEqual(result['mean_foreground_guidance_coverage'],1.)
            self.assertEqual(set(result['calibration_label_sha256']),set(names))
            for cells in result['details'].values():
                self.assertTrue(cells['cells']['2:0']['enabled'])
                self.assertEqual(cells['cells']['2:0']['cases'],5)
                self.assertEqual(cells['cells']['2:0']['pixels'],5*4*16*16)
                self.assertFalse(cells['cells']['2:1']['enabled'])

    def test_active_competence_engine_updates_without_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate=fake_gate(Path(tmp)/'gate.json')
            cfg=small_config(method='mean_teacher',prompt_source='text',prompt_cache='/unused',
                prompt_route='competence',prompt_gate=str(gate),confidence=.99,background_confidence=.99)
            model=StudentTeacherSystem(cfg);batch=synthetic_batch()
            batch['prompt_targets']=torch.ones(4,4,32,32)
            metrics=train_step(model,make_optimizer(model,cfg),batch,cfg,torch.device('cpu'),601)
            self.assertGreater(metrics['tissue_coverage_s1'],0)
            self.assertEqual(metrics['fine_tissue_overlap_s1'],0)
            self.assertGreater(metrics['weighted_tissue_ce_s1'],0)

    def test_v2_export_image_only_and_reader_roundtrip(self):
        from prompt_teacher.revision import export
        from freqssl.anatomy import FAMILY_IDS
        class FakeModel:
            pass
        def predict(model,views,images,seed):
            q=torch.zeros(len(images),4,4,224,224);q[:,:,1]=1
            return q
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);png=make_spider_fixture(tmp)
            cfg=small_config(png,method='mean_teacher',image_size=224)
            _,_,split=build_training_data(cfg)
            checkpoint=tmp/'teacher.pt';checkpoint.write_bytes(b'fixture')
            meta={'mean':[0.,0.,0.],'std':[1.,1.,1.]}
            for n in split['unlabeled_files']:(png/'train/label'/n).unlink()
            with mock.patch('prompt_teacher.revision.load_revision',return_value=(FakeModel(),meta,None,{'step':500},split)), \
                 mock.patch('prompt_teacher.revision.predict_templates',side_effect=predict), \
                 mock.patch('torch.cuda.reset_peak_memory_stats'),mock.patch('torch.cuda.max_memory_allocated',return_value=0):
                export(cfg,'text','T1',checkpoint,tmp/'cache',torch.device('cpu'))
            cache=PromptCache(tmp/'cache','text',224,split)
            targets=unpack_template_probabilities(cache.read(split['unlabeled_files'][0])[None])
            self.assertTrue((targets[:,0]==1).all());self.assertTrue(targets[:,1:3].bool().all())

    def test_preserve_ema_keeps_confident_correct_number_even_when_teacher_conflicts(self):
        logits=torch.randn(1,20,2,2,requires_grad=True);ema=torch.zeros_like(logits);ema[:,1]=.99;ema[:,0]=.01
        tissue=torch.full((1,2,2),3);mask=torch.ones_like(tissue,dtype=torch.bool)
        original,_=masked_pseudo_loss(logits,ema,.8,.95,.25)
        actual,metrics=prompt_routed_loss(logits,ema,tissue,mask,mask,route='preserve_ema')
        torch.testing.assert_close(actual,original,rtol=0,atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual,logits,retain_graph=True)[0],torch.autograd.grad(original,logits)[0],rtol=0,atol=0)
        self.assertEqual(metrics['tissue_coverage'],0)
        self.assertEqual(metrics['fine_coverage'],1)

    def test_number_uncertainty_does_not_imply_tissue_uncertainty(self):
        ema=torch.zeros(1,20,1,2);ema[:,1:10,:,0]=1/9;ema[:,:,:,1]=1/20
        t=torch.ones(1,1,2,dtype=torch.long);m=torch.ones_like(t,dtype=torch.bool)
        eligible=competence_eligible(ema,t,m,m)
        self.assertEqual(eligible.tolist(),[[[False,True]]])
        loss,metrics=prompt_routed_loss(torch.zeros_like(ema,requires_grad=True),ema,t,m,m,route='competence',gate_mask=m)
        self.assertEqual(metrics['tissue_coverage'],.5)
        self.assertEqual(metrics['fine_tissue_overlap'],0)
        self.assertAlmostEqual(loss.item(),.5*-.5*np.log(9/20),places=6)

    def test_gate_closed_exactly_recovers_original_pseudo_loss(self):
        logits=torch.randn(2,20,3,4,requires_grad=True);ema=torch.randn_like(logits).mul(5).softmax(1)
        t=torch.ones(2,3,4,dtype=torch.long);m=t.bool()
        expected,_=masked_pseudo_loss(logits,ema,.8,.95,.25)
        actual,_=prompt_routed_loss(logits,ema,t,m,m,route='competence',gate_mask=~m)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_probability_interpolation_renormalization_padding_and_edges(self):
        q=torch.zeros(1,4,4,8,8);q[:,:,1,:,:4]=1;q[:,:,3,:,4:]=1
        packed=torch.cat([q.flatten(1,2),torch.ones(1,1,8,8)],1)
        out=unpack_template_probabilities(packed)
        self.assertTrue(out[:,1:3].bool().all());self.assertEqual(out[0,3,3].tolist(),[0,0,1,1,1,1,0,0])
        packed[:,:,:,:2]=0;packed[:,:16,:,2]=packed[:,:16,:,2]*.99;packed[:,16,:,2]=.99
        out=unpack_template_probabilities(packed)
        self.assertFalse(out[:,1:3,:,:3].bool().any())
        self.assertTrue(out[:,1:3,:,3:].bool().all())

    def test_fixed_case_partition_excludes_partial_case_and_allows_no_expansion(self):
        labeled=[f'SPIDER_{i:03d}_0_1.png' for i in range(1,69)]+['SPIDER_069_0_2.png']
        unlabeled=['SPIDER_069_1_2.png','SPIDER_070_0_1.png']
        p=make_partition(dict(labeled_files=labeled,unlabeled_files=unlabeled))
        self.assertEqual(len(p['calibration_cases']),14);self.assertNotIn('SPIDER_069',p['calibration_cases'])
        self.assertFalse(set(p['calibration_files'])&set(p['labeled_files']+p['unlabeled_files']))
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'partition.json';f.write_text(json.dumps(p));self.assertEqual(read_partition(f,labeled,unlabeled),p)
            p['labeled_files'].append(unlabeled[0]);f.write_text(json.dumps(p))
            with self.assertRaises(ValueError):read_partition(f,labeled,unlabeled)

    def test_gate_phase_limits_warmup_missing_cells_and_provenance(self):
        self.assertEqual([gate_stage(s) for s in [1,600,601,1000,1001,6000]],[500,500,500,500,1000,5500])
        with tempfile.TemporaryDirectory() as tmp:
            p=fake_gate(Path(tmp)/'gate.json');g=CompetenceGate(p,'text');t=torch.ones(1,3,4,dtype=torch.long)
            self.assertFalse(g.lookup(t,t,600).any());self.assertTrue(g.lookup(t,t,601).all())
            with self.assertRaises(ValueError):CompetenceGate(p,'neutral')
            v=json.loads(p.read_text());del v['tables']['1000'];p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):CompetenceGate(p,'text')

    def test_capability_counts_cases_instead_of_treating_pixels_as_independent(self):
        self.assertFalse(qualify_cell([[100000,100000,0]],1)['enabled'])
        self.assertTrue(qualify_cell([[300,297,240]]*5,1)['enabled'])
        self.assertFalse(qualify_cell([[300,285,285]]*5,1)['enabled'])
        self.assertFalse(qualify_cell([[300,285,240]]*5,0)['enabled'])

    def test_js_zero_for_equal_predictions_and_nonzero_with_gradients(self):
        a=torch.randn(2,4,8,8,requires_grad=True);gt=torch.randint(4,(2,8,8))
        self.assertLess(abs(balanced_js(a,a,gt).item()),1e-7)
        b=-a;value=balanced_js(a,b,gt);value.backward()
        self.assertGreater(value.item(),0);self.assertGreater(a.grad.abs().sum().item(),0)

    def test_teacher_selection_requires_every_gate_and_prefers_simpler(self):
        scores=dict(dice=.8,tissue_dice={'vertebra':.85,'canal':.85,'disc':.7})
        control={'canonical':scores};candidate={k:scores for k in ['canonical','heldout_4','heldout_5']}
        gate={'tables':{'500':[[False]*2,[True,False],[False]*2,[False]*2]},'mean_foreground_guidance_coverage':.002}
        passed=teacher_screen(control,candidate,gate);self.assertTrue(passed['passed'])
        self.assertFalse(teacher_screen(control,candidate,dict(gate,mean_foreground_guidance_coverage=0))['passed'])
        self.assertEqual(choose_teacher({'T1':passed,'T2':passed},{'T1':.8,'T2':.8005}),'T1')
        self.assertEqual(choose_teacher({'T1':passed,'T2':passed},{'T1':.8,'T2':.802}),'T2')

    def test_v2_cache_training_resume_and_teacher_free_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);png=make_spider_fixture(tmp);base=small_config(png,method='mean_teacher',max_steps=3,eval_every=3)
            original,_,split=build_training_data(base);cache=fake_v2(tmp/'cache',split);gate=fake_gate(tmp/'gate.json')
            cfg=replace(base,prompt_source='text',prompt_cache=str(cache),prompt_route='competence',prompt_gate=str(gate),output_dir=str(tmp/'whole'),checkpoint_every=1)
            ds,_,_=build_training_data(cfg)
            for n in split['unlabeled_files']:(png/'train/label'/n).unlink()
            for epoch in [0,1,2]:
                ds.set_epoch(epoch);original.set_epoch(epoch)
                for i in range(len(ds)):
                    a,b=ds[i],original[i];self.assertTrue(torch.equal(a['image'],b['image']));self.assertEqual(a['prompt_targets'].shape,(4,32,32))
            run_experiment(cfg,'cpu');second=replace(cfg,output_dir=str(tmp/'resumed'))
            run_experiment(second,'cpu',stop_after=1);run_experiment(second,'cpu',resume=tmp/'resumed/last.pt')
            a=torch.load(tmp/'whole/last.pt',weights_only=True);b=torch.load(tmp/'resumed/last.pt',weights_only=True)
            for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]),k)
            self.assertTrue((tmp/'whole/step_2.pt').exists());cache.rename(tmp/'removed');gate.unlink()
            self.assertIsNotNone(load_single_network(tmp/'whole/last.pt',torch.device('cpu')))
