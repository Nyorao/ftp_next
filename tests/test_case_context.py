"""Reference/position controls; all training here is tiny synthetic smoke testing."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import hashlib
import SimpleITK as sitk
from PIL import Image

from freqssl.augment import strong_view, aligned_cutmix
from freqssl.case_context import CaseContextUNet, reference_index, relative_position
from freqssl.context_cache import iter_cached_logits
from freqssl.data import build_training_data, SpiderSliceDataset
from freqssl.engine import train_step, make_optimizer, state_hash, seed_everything, run_experiment
from freqssl.inference import load_single_network
from freqssl.models import StudentTeacherSystem
from test_freqssl import CPUInvariantTest, small_config, synthetic_batch, make_spider_fixture


def cfg_for(arm, **kwargs):
    return small_config(method='mean_teacher',slice_context='adaptive',
        case_reference='middle' if arm=='B' else 'none',
        context_position='relative' if arm=='C' else 'none',
        context_geometry_manifest='/unused-geometry.json' if arm=='C' else '',**kwargs)


def fake_geometry(dataset,path):
    from spider_volume import group_complete_slices
    positions={}
    for case,names in group_complete_slices(dataset.names).items():
        positions.update({name:relative_position(i,len(names),True) for i,name in enumerate(names)})
    return positions,{'annotation_files_read':0,'test_fixture':True,
        'cases':{case:{'spacing_mm':4.} for case in group_complete_slices(dataset.names)}}


class CaseContextTests(CPUInvariantTest):
    def test_initialization_parameters_and_missing_position(self):
        models={a:StudentTeacherSystem(cfg_for(a)).students[0] for a in ('A','B','C')}
        for name,value in models['A'].state_dict().items():
            if not name.startswith('context_fusion'):
                self.assertTrue(torch.equal(value,models['B'].state_dict()[name]))
            self.assertTrue(torch.equal(value,models['C'].state_dict()[name]))
        x=torch.randn(2,3,32,32)
        pos=torch.zeros(2,1,32,32)
        for layer in models['A'].context_fusion:
            layer.gamma.data.fill_(.4)
        models['C'].load_state_dict(models['A'].state_dict(),strict=False)
        torch.testing.assert_close(models['A'](x),models['C'](x,pos),rtol=0,atol=0)
        with self.assertRaises(ValueError):
            models['C'](x)
        large={a:StudentTeacherSystem(replace(cfg_for(a),base_channels=32,max_channels=256)).students[0] for a in ('A','B','C')}
        n={a:sum(p.numel() for p in m.parameters()) for a,m in large.items()}
        self.assertEqual(n['B']-n['A'],4483)
        self.assertEqual(n['C']-n['A'],249)
        with self.assertRaises(ValueError):
            replace(cfg_for('B'),context_position='relative').validate()
        with self.assertRaises(ValueError):
            replace(cfg_for('C'),context_geometry_manifest='').validate()

    def test_indices_geometry_and_dataset_annotation_boundary(self):
        self.assertEqual([reference_index(n) for n in (1,2,5,6)],[0,0,2,2])
        self.assertEqual(relative_position(0,1),0)
        self.assertEqual(relative_position(0,5,True),1)
        self.assertEqual(relative_position(4,5,True),-1)
        with tempfile.TemporaryDirectory() as tmp:
            root=make_spider_fixture(Path(tmp))
            a,_,split=build_training_data(cfg_for('A',root=root,labelnum=3,max_samples=8))
            b,_,other=build_training_data(cfg_for('B',root=root,labelnum=3,max_samples=8))
            self.assertEqual(split['labeled_files'],other['labeled_files'])
            from PIL import Image
            for name in b.names[3:]:
                (root/'train/label'/name).unlink()
            for i in range(8):
                with mock.patch('PIL.Image.open',wraps=Image.open) as opened:
                    sample=b[i]
                labels=[Path(c.args[0]).name for c in opened.call_args_list if Path(c.args[0]).parent.name=='label']
                self.assertEqual(labels,[b.names[i]] if i<3 else [])
                torch.testing.assert_close(sample['image'][:3],a[i]['image'],rtol=0,atol=0)
                self.assertEqual(sample['image'].shape,(4,32,32))
                self.assertEqual(b.references[b.names[i]],b.names[(i//4)*4+1])
            with mock.patch('freqssl.case_context.verify_dataset_geometry',side_effect=fake_geometry):
                c,_,_=build_training_data(cfg_for('C',root=root,labelnum=3,max_samples=8))
            self.assertEqual(float(c[0]['position'][0,0,0]),1.)
            with self.assertRaises(FileNotFoundError):
                build_training_data(cfg_for('C',root=root,labelnum=3,max_samples=8))

    def test_four_channel_random_stream_and_position_cutmix(self):
        x=torch.randn(4,4,32,32)
        torch.testing.assert_close(strong_view(x,7,.08,.15)[:,:3],strong_view(x[:,:3],7,.08,.15),rtol=0,atol=0)
        ids=torch.linspace(-1,1,4)[:,None,None,None].expand(4,4,32,32).clone()
        pseudo=torch.zeros(4,20,32,32)
        mixed,packed=aligned_cutmix(ids,torch.cat((pseudo,ids[:,:1]),1),91,1.)
        for channel in range(4):
            self.assertTrue(torch.equal(mixed[:,channel:channel+1],packed[:,20:]))

    def test_verified_geometry_uses_only_images_and_rejects_changed_bytes(self):
        from freqssl.inference import image_geometry,legacy_gray8
        from freqssl.geometry import sagittal_plan,canonicalize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'train/image').mkdir(parents=True)
            array=np.random.default_rng(3).integers(0,1000,(29,27,4),dtype=np.int16)
            image=sitk.GetImageFromArray(array);image.SetDirection((-1.,0.,0.,0.,1.,0.,0.,0.,-1.))
            source=root/'SPIDER_1.mha';sitk.WriteImage(image,str(source))
            geo=image_geometry(image);plan=sagittal_plan(geo)
            ordered=canonicalize(array,plan)
            if 0 in plan['flip_canonical_axes']:
                ordered=ordered[::-1]
            for i,plane in enumerate(ordered):
                Image.fromarray(legacy_gray8(plane).T).save(root/f'train/image/SPIDER_1_{i}_4.png')
            record=dict(case='SPIDER_1',split='train',converted_image=str(source),verified_geometry=geo,
                converted_image_array_sha256=hashlib.sha256(array.tobytes()).hexdigest())
            manifest=root/'geometry.json';manifest.write_text(json.dumps({'records':[record]}))
            data=SpiderSliceDataset(root,'train',32,labeled_names=[],slice_context='adaptive',
                context_position='relative',context_geometry_manifest=str(manifest))
            np.testing.assert_allclose([data.positions[n] for n in data.names],[1.,1/3,-1/3,-1.])
            self.assertTrue((data[0]['label']==-100).all())
            record['converted_image_array_sha256']='wrong'
            manifest.write_text(json.dumps({'records':[record]}))
            with self.assertRaisesRegex(ValueError,'hash'):
                SpiderSliceDataset(root,'train',32,labeled_names=[],slice_context='adaptive',
                    context_position='relative',context_geometry_manifest=str(manifest))

    def test_candidate_gates_and_intervention_shuffle(self):
        from experiments.run_case_reference import choose_candidate,confirmed
        from experiments.analyze_case_reference import shuffled_positions
        def row(d,u): return {'dice':d,'dice_union_present':u}
        self.assertIsNone(choose_candidate({'A':row(.8,.77),'B':row(.81,.76),'C':row(.802,.78)}))
        self.assertEqual(choose_candidate({'A':row(.8,.77),'B':row(.81,.78),'C':row(.81,.78)}),'C')
        self.assertEqual(choose_candidate({'A':row(.8,.77),'B':row(.82,.78),'C':row(.81,.78)}),'B')
        self.assertTrue(confirmed([.5,.6,-.1],[0,0,0]))
        self.assertFalse(confirmed([.5,.5,.5],[0,0,-.1]))
        self.assertFalse(confirmed([.5],[0]))
        p=[-1.,-.5,0,.5,1.]
        self.assertEqual(shuffled_positions(p,'case'),shuffled_positions(p,'case'))
        self.assertCountEqual(shuffled_positions(p,'case'),p)
        self.assertNotEqual(shuffled_positions(p,'case'),p)
        self.assertEqual(shuffled_positions([0.],'case'),[0.])

    def test_analysis_complete_native_metrics_pairs_and_interventions(self):
        from experiments.analyze_case_reference import analyze,paired_analysis,combine_seed_pairs,confusion_metrics
        from freqssl.evaluation import prepare_validation,evaluate_volumes
        cm=np.zeros((20,20),dtype=np.int64);cm[1,1]=8;cm[1,2]=2;cm[0,3]=3
        scores=confusion_metrics(cm)
        self.assertAlmostEqual(scores['dice_percent'],800/9)
        self.assertEqual(scores['same_tissue_wrong_level_percent'],20.)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch('freqssl.case_context.verify_dataset_geometry',side_effect=fake_geometry), \
                mock.patch('experiments.analyze_case_reference.verify_dataset_geometry',side_effect=fake_geometry):
            root=Path(tmp);data=make_spider_fixture(root/'data');summaries={}
            for arm in ('A','B','C'):
                cfg=cfg_for(arm,root=data,output=root/arm,labelnum=4,max_samples=8)
                system=StudentTeacherSystem(cfg)
                checkpoint=root/f'{arm}.pt'
                torch.save({'format_version':3,'config':cfg.to_dict(),'model':system.state_dict(),'step':0},checkpoint)
                dataset,records=prepare_validation(cfg)
                native=evaluate_volumes(system,dataset,records,'cpu')['teacher1']['overall']
                summary=analyze(checkpoint,root/f'analysis_{arm}','cpu',len(records))
                summaries[arm]=summary
                self.assertEqual(summary['metrics']['full']['overall'],native)
                self.assertAlmostEqual(summary['regions']['all']['full']['native_dice_percent']['mean'],100*native['dice'])
                self.assertEqual(len(summary['conditions']),{'A':1,'B':3,'C':2}[arm])
                self.assertIn('reference_duplicate',summary['regions'])
                self.assertIn('boundary_background_absent_slice_class_pixels_percent',summary['regions']['all']['full'])
                self.assertEqual(summary['training_updates'],0)
            pair=paired_analysis(summaries['B'],summaries['A'])
            combined=combine_seed_pairs({str(s):pair for s in (42,43,44)})
            self.assertEqual(combined['all']['native_dice_percent']['n_cases'],len(records))
            self.assertEqual(len(combined['all']['native_dice_percent']['per_seed']),3)
            with self.assertRaisesRegex(ValueError,'complete'):
                analyze(checkpoint,root/'incomplete','cpu',21)
            self.assertEqual(json.loads((root/'incomplete/status.json').read_text())['status'],'failed')

    def test_native_image_only_inference_preserves_acquisition_and_position(self):
        from freqssl.geometry import sagittal_plan,canonicalize,restore_native
        from freqssl.inference import SingleNetwork,predict_volume,prepare_slice
        from torch.nn import functional as F
        array=np.random.default_rng(17).integers(0,2000,(31,29,4),dtype=np.int16)
        for reverse in (False,True):
            plan=sagittal_plan({'size_xyz':[4,29,31],'spacing_xyz_mm':[4.,.8,.8],
                'direction':[-1. if reverse else 1.,0.,0.,0.,1.,0.,0.,0.,-1.]})
            canonical=canonicalize(array,plan);ordered=canonical[::-1] if reverse else canonical
            images=torch.stack([prepare_slice(v,32) for v in ordered]);n=len(images)
            for arm in ('B','C'):
                backbone=StudentTeacherSystem(cfg_for(arm)).students[0].eval().requires_grad_(False)
                for fusion in backbone.context_fusion:
                    fusion.gamma.fill_(.5);fusion.weights.weight.normal_(0,.1)
                    if arm=='C': fusion.position_bias[-1].weight.normal_(0,.1)
                indices=[[max(0,i-1),i,min(n-1,i+1)]+([reference_index(n)] if arm=='B' else []) for i in range(n)]
                position=torch.tensor([relative_position(i,n,reverse) for i in range(n)]).reshape(n,1,1,1)
                with torch.no_grad():
                    logits=backbone(torch.stack([images[ix,0] for ix in indices]),position if arm=='C' else None)
                expected=F.interpolate(logits.argmax(1).float()[:,None],size=canonical.shape[1:],mode='nearest')[:,0].numpy().astype(np.uint8)
                if reverse: expected=expected[::-1]
                actual=predict_volume(SingleNetwork(backbone,0.),array,plan,32,'cpu',3)
                np.testing.assert_array_equal(actual,restore_native(expected,plan))

    def test_prepare_freezes_three_arms_without_starting_training(self):
        from experiments.run_case_reference import prepare,sha,ROOT
        from freqssl.config import load_config
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);geometry=root/'geometry.json'
            geometry.write_text(json.dumps({'records':[]}))
            base=replace(load_config(ROOT/'configs/v3/case_reference.yaml'),context_geometry_manifest=str(geometry))
            config=root/'base.json';config.write_text(json.dumps(base.to_dict()))
            args=SimpleNamespace(output_dir=root/'prepared',config=config,device='cuda:0',screen_only=False)
            with mock.patch('experiments.run_case_reference.architecture_profiles',return_value={}), \
                    mock.patch('experiments.run_case_reference.run_experiment') as training, \
                    mock.patch('experiments.run_case_reference.subprocess.Popen') as launch:
                command=prepare(args)
            training.assert_not_called();launch.assert_not_called()
            self.assertIn('--worker',command)
            self.assertEqual(json.loads((args.output_dir/'status.json').read_text()),{'status':'prepared','training_started':False})
            manifest=json.loads((args.output_dir/'manifest.json').read_text())
            self.assertTrue(manifest['confirm_on_pass'])
            for name,digest in manifest['source_hashes'].items():
                self.assertEqual(sha(args.output_dir/'code'/name),digest)
            for arm,(ref,pos) in {'A':('none','none'),'B':('middle','none'),'C':('none','relative')}.items():
                cfg=manifest['configs'][arm]
                self.assertEqual((cfg['case_reference'],cfg['context_position']),(ref,pos))
                self.assertEqual((cfg['seed'],cfg['max_steps'],cfg['eval_every']),(42,6000,500))
            with self.assertRaises(FileExistsError): prepare(args)

    def test_engine_passes_mixed_position_and_updates_ema(self):
        cfg=replace(cfg_for('C'),cutmix_probability=1.)
        seed_everything(cfg)
        model=StudentTeacherSystem(cfg)
        batch=synthetic_batch();batch['image']=torch.randn(4,3,32,32)
        batch['position']=torch.linspace(-1,1,4)[:,None,None,None].expand(4,1,32,32).clone()
        teacher_calls,student_calls=[],[]
        original_teacher,original_student=model.teacher_logits,model.student_logits
        def teacher(x,index,position=None):
            teacher_calls.append(position.clone())
            return original_teacher(x,index,position)
        def student(x,index,position=None):
            student_calls.append(position.clone())
            return original_student(x,index,position)
        before=state_hash(model.students[0])
        with mock.patch.object(model,'teacher_logits',side_effect=teacher), mock.patch.object(model,'student_logits',side_effect=student):
            train_step(model,make_optimizer(model,cfg),batch,cfg,torch.device('cpu'),0)
        self.assertNotEqual(before,state_hash(model.students[0]))
        self.assertSameState(model.students[0],model.teachers[0])
        expected=aligned_cutmix(batch['image'][2:],batch['position'][2:],cfg.seed+17,1.)[1]
        torch.testing.assert_close(student_calls[0][2:],expected,rtol=0,atol=0)
        torch.testing.assert_close(teacher_calls[0],batch['position'][2:],rtol=0,atol=0)
        for arm in ('B','C'):
            m=StudentTeacherSystem(cfg_for(arm)).students[0]
            for f in m.context_fusion:
                f.gamma.data.fill_(.5)
            x=torch.randn(2,4 if arm=='B' else 3,32,32,requires_grad=True)
            p=torch.randn(2,1,32,32).tanh() if arm=='C' else None
            m(x,p).square().mean().backward()
            self.assertGreater(float(x.grad[:,0].abs().sum()),0)
            if arm=='B':
                self.assertGreater(float(x.grad[:,3].abs().sum()),0)
            else:
                self.assertGreater(float(m.context_fusion[0].position_bias[-1].weight.grad.abs().sum()),0)

    def check_cache(self,device):
        for arm in ('B','C'):
            m=StudentTeacherSystem(cfg_for(arm)).students[0].to(device).eval().requires_grad_(False)
            for f in m.context_fusion:
                f.gamma.fill_(.6);f.weights.weight.normal_(0,.1)
                if arm=='C':
                    f.position_bias[-1].weight.normal_(0,.1)
            for n in (1,2,9,18):
                images=torch.randn(n,1,32,32)
                p=torch.tensor([relative_position(i,n,True) for i in range(n)],device=device).reshape(n,1,1,1)
                for policy in (('middle','target','quarter') if arm=='B' else ('middle',)):
                    selections=[[max(0,i-1),i,min(n-1,i+1)]+([reference_index(n,policy,i)] if arm=='B' else []) for i in range(n)]
                    with torch.no_grad():
                        expected=m(torch.stack([images[ix,0] for ix in selections]).to(device),p if arm=='C' else None)
                    for batch in (1,8,11):
                        reads,stats=[],{}
                        def read(i):
                            reads.append(i);return images[i]
                        kwargs={'read_position':lambda i:float(p[i,0,0,0])} if arm=='C' else {}
                        actual=torch.cat([v for _,v in iter_cached_logits(m,n,read,device,batch,
                            reference_policy=policy,stats=stats,**kwargs)])
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0 if device=='cpu' else 1e-5)
                        self.assertTrue(torch.equal(actual.argmax(1),expected.argmax(1)))
                        self.assertEqual(sorted(reads),list(range(n)))
                        self.assertEqual(stats['real_slices_encoded'],n)

    def test_cache_cpu(self):
        self.check_cache('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cache_cuda(self):
        self.check_cache('cuda:0')

    def test_checkpoint_resume_both_arms(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('freqssl.case_context.verify_dataset_geometry',side_effect=fake_geometry):
            root=Path(tmp);data=make_spider_fixture(root/'data')
            for arm in ('B','C'):
                cfg=cfg_for(arm,root=data,output=root/(arm+'_full'),labelnum=4,max_samples=8)
                full=run_experiment(cfg,'cpu')
                staged=replace(cfg,output_dir=str(root/(arm+'_staged')))
                run_experiment(staged,'cpu',stop_after=1)
                resumed=run_experiment(staged,'cpu',resume=Path(staged.output_dir)/'last.pt')
                self.assertEqual(full['final_teacher1_hash'],resumed['final_teacher1_hash'])
                self.assertEqual(full['final_validation'],resumed['final_validation'])
                single,saved,receipt=load_single_network(Path(staged.output_dir)/'best.pt','cpu')
                self.assertIsInstance(single.backbone,CaseContextUNet)
                self.assertEqual(receipt['case_reference'],cfg.case_reference)
                if arm=='B':
                    with self.assertRaises(RuntimeError):
                        StudentTeacherSystem(cfg_for('A')).students[0].load_state_dict(single.backbone.state_dict())


if __name__=='__main__':
    unittest.main()
