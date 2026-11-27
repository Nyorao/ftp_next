"""Synthetic invariants for reference conditioning, not efficacy experiments."""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import json
from unittest import mock
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from freqssl.augment import aligned_cutmix
from freqssl.config import ExperimentConfig
from freqssl.data import build_training_data, SpiderSliceDataset
from freqssl.engine import make_optimizer, train_step, run_experiment, state_hash
from freqssl.evaluation import prepare_validation, evaluate_volumes
from freqssl.inference import load_single_network
from freqssl.models import StudentTeacherSystem
from freqssl.reference_conditioning import RowWindowAttention, si_coordinates, row_prior
from freqssl.reference_data import (build_prediction_cache, ReferencePredictionCache, ReferenceTrainingDataset,
    candidate_references, select_reference, file_hash)
from freqssl.reference_inference import iter_reference_logits
from freqssl.reference_training import gt_probability, soft_cycle_loss
from freqssl.reference_metrics import numbering_metrics, WorkCounter
from test_freqssl import CPUInvariantTest, small_config, make_spider_fixture
from test_case_context import fake_geometry


def reference_config(**kwargs):
    return small_config(method='mean_teacher', slice_context='adaptive', reference_condition='kv',
        reference_teacher='/fixture/A.pt', context_geometry_manifest='/fixture/geometry.json', **kwargs)


def reference_batch(b=4, shape=(32, 32)):
    return dict(image=torch.randn(b, 1, *shape), probabilities=torch.randn(b, 20, *shape).softmax(1),
                si=si_coordinates(b, shape), valid=torch.ones(b, 1, *shape))


def fixture(root, condition='kv', gt=False, cycle=0.):
    data = make_spider_fixture(root / 'data')
    base = small_config(root=data, method='mean_teacher', slice_context='adaptive', labelnum=3, max_samples=8)
    source = root / 'A.pt'
    torch.save(dict(format_version=3, config=base.to_dict(), model=StudentTeacherSystem(base).state_dict(), step=2), source)
    training, _, _ = build_training_data(base)
    validation, _ = prepare_validation(base)
    cache = root / 'cache'; cache.mkdir()
    for split, dataset in (('train', training), ('valid', validation)):
        build_prediction_cache(source, dataset, cache / split)
    cfg = replace(base, output_dir=str(root / 'run'), reference_condition=condition,
        reference_teacher=str(source), reference_teacher_sha256=file_hash(source), reference_cache=str(cache),
        context_geometry_manifest='/fixture/geometry.json', reference_gt=gt, reference_cycle_weight=cycle).validate()
    return cfg


class ReferenceModelTests(CPUInvariantTest):
    def test_common_initialization_zero_gate_and_gradient_start(self):
        cfg = reference_config()
        plain = StudentTeacherSystem(replace(cfg, reference_condition='none', reference_teacher='')).students[0]
        state = torch.get_rng_state().clone()
        model = StudentTeacherSystem(cfg).students[0]
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        for a, b in zip(plain.context_fusion, model.context_fusion):
            a.gamma.data.fill_(.3); b.gamma.data.fill_(.3)
        images, ref = torch.randn(2, 3, 32, 32), reference_batch(2)
        expected = plain(images)
        actual = model(images, reference=ref)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertTrue(all(m.gate.grad.abs() > 0 for m in model.reference_adapters))
        self.assertTrue(all(m.query.weight.grad is None or m.query.weight.grad.count_nonzero() == 0 for m in model.reference_adapters))
        model.zero_grad()
        for adapter in model.reference_adapters:
            adapter.gate.data.fill_(.1)
        model(images, reference=ref).square().mean().backward()
        self.assertTrue(all(m.query.weight.grad.abs().sum() > 0 for m in model.reference_adapters))
        self.assertGreater(model.mask_encoder.layers[0][0].weight.grad.abs().sum(), 0)
        self.assertEqual(model.encoders[0][0].in_channels, 1)
        self.assertEqual(model.mask_encoder.layers[0][0].in_channels, 21)

    def test_sparse_attention_matches_dense_and_rejects_padding(self):
        module = RowWindowAttention(8, 1, dimension=16).double()
        module.gate.data.fill_(.4)
        q, k, v = [torch.randn(2, 8, 3, 5, dtype=torch.float64) for _ in range(3)]
        si = si_coordinates(2, (3, 5)).double()
        valid = torch.ones_like(si); valid[1] = 0
        sparse = module(q, k, v, si, si, valid, valid)
        dense = module(q, k, v, si, si, valid, valid, dense=True)
        torch.testing.assert_close(sparse, dense, atol=1e-9, rtol=1e-9)
        self.assertEqual(sparse[1].count_nonzero(), 0)
        module.observe = True
        module(q, k, v, si, si, valid, valid)
        self.assertEqual(module.observations[-1]['queries'], 0)
        self.assertAlmostEqual(sum(module.observations[0]['mass'][0]), 15, places=5)
        self.assertGreater(module.attention_macs, 0)

    def test_position_keys_ignore_reference_appearance_only(self):
        module = RowWindowAttention(8, 1, position_only=True)
        module.gate.data.fill_(.5)
        q, k, v = [torch.randn(1, 8, 3, 5) for _ in range(3)]
        si = si_coordinates(1, (3, 5)); valid = torch.ones_like(si)
        first = module(q, k, v, si, si, valid, valid)
        torch.testing.assert_close(first, module(q, k * 30 + 10, v, si, si, valid, valid), rtol=0, atol=0)
        self.assertFalse(torch.equal(first, module(q, k, v * 2, si, si, valid, valid)))

    def test_si_axis_and_row_prior_on_non_square_plane(self):
        p = torch.zeros(1, 20, 32, 48); p[:, 2, :, :24] = 1; p[:, 3, :, 24:] = 1
        profile = row_prior(p)
        torch.testing.assert_close(profile, p)
        coords = si_coordinates(1, (32, 48))
        self.assertTrue(torch.equal(coords[:, :, 0], coords[:, :, -1]))
        self.assertLess(float(coords[0, 0, 0, 0]), float(coords[0, 0, 0, -1]))

    def test_row_profile_precedes_shared_rotation_and_preserves_si_coordinates(self):
        from freqssl.reference_data import group_names
        with tempfile.TemporaryDirectory() as tmp:
            root = make_spider_fixture(Path(tmp))
            base = SpiderSliceDataset(root, 'train', (32,48), augment=True, seed=0, slice_context='adaptive')
            groups = group_names(base.names)
            p = torch.zeros(20,32,48); p[1,:,:24]=1; p[2,:,24:]=1
            cache = SimpleNamespace(groups=groups, selected={c:1 for c in groups},
                candidates={c:[1] for c in groups}, read=lambda name:p)
            cfg = replace(reference_config(), reference_condition='row', reference_sampling='fixed')
            data = ReferenceTrainingDataset(base, cfg, cache)
            sample = data[0]
            si = si_coordinates(1,(32,48))[0]; valid=torch.ones_like(si)
            packed = torch.cat((row_prior(p[None])[0],si,valid))
            _,_,warped = base._geometry(torch.zeros(1,32,48),None,0,packed,'bilinear')
            expected = warped / warped[-1:].clamp_min(1e-8)
            torch.testing.assert_close(sample['reference_prior'], expected[:20], rtol=0,atol=0)
            torch.testing.assert_close(sample['reference_si'],expected[20:21],rtol=0,atol=0)
            self.assertFalse(torch.allclose(sample['reference_prior'],row_prior(sample['reference_probabilities'][None])[0]))

    def test_cutmix_routing_matches_original_and_isolates_reference_banks(self):
        images = torch.randn(4, 3, 32, 32); p = torch.randn(4, 20, 32, 32).softmax(1)
        original = aligned_cutmix(images, p, 71, 1.)
        mixed, target, donor, mask = aligned_cutmix(images, p, 71, 1., return_routing=True)
        torch.testing.assert_close(mixed, original[0], rtol=0, atol=0)
        torch.testing.assert_close(target, original[1], rtol=0, atol=0)
        torch.testing.assert_close(mixed, images * (1-mask) + images[donor] * mask)
        model = StudentTeacherSystem(reference_config()).students[0]
        for adapter in model.reference_adapters:
            adapter.gate.data.fill_(.5)
        ref = reference_batch(2)
        bank = model.prepare_reference(**dict(image=ref['image'], probabilities=ref['probabilities'], si=ref['si'], valid=ref['valid']))
        encoded = model.encode(torch.randn(6, 1, 32, 32))
        triplets = [v.reshape(2, 3, *v.shape[1:]) for v in encoded]
        full = torch.zeros(2, 1, 32, 32); full[0] = 1
        donors = torch.tensor([1, 0])
        # Inspect the injection itself, before spatial decoder convolutions.
        with mock.patch.object(model, 'decode_features', side_effect=lambda skips: skips[3]):
            first = model.decode_reference(triplets, bank, ref['si'], ref['valid'], donors, full)
            changed = [{k: None if v is None else v.clone() for k, v in level.items()} for level in bank]
            for level in changed:
                level['value'][0] += 100
            second = model.decode_reference(triplets, changed, ref['si'], ref['valid'], donors, full)
        # Sample 0 is wholly donor 1; sample 1 is wholly its own bank 1.
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_cached_inference_matches_direct_with_nonzero_gates(self):
        for condition in ('row', 'kv', 'position'):
            cfg = replace(reference_config(), reference_condition=condition)
            model = StudentTeacherSystem(cfg).students[0].eval()
            if condition == 'row':
                for a in model.reference_adapters:
                    torch.nn.init.normal_(a.adapter[-1].weight, std=.02)
            else:
                for a in model.reference_adapters:
                    a.gate.data.fill_(.3)
            images = torch.randn(5, 1, 32, 32)
            p = torch.randn(5, 20, 32, 32).softmax(1)
            expected_input = torch.cat([images[[max(0, i-1), i, min(4, i+1)]].transpose(0, 1) for i in range(5)])
            ref = dict(image=images[2:3].expand(5, -1, -1, -1), probabilities=p[2:3].expand(5, -1, -1, -1),
                si=si_coordinates(5, (32, 32)), valid=torch.ones(5, 1, 32, 32))
            with torch.no_grad():
                direct = model(expected_input, reference=ref)
                stats = {}
                cached = torch.cat([v for _, v in iter_reference_logits(model, 5, lambda i: images[i], lambda i: p[i],
                    'cpu', 2, reference_index=2, stats=stats)])
            torch.testing.assert_close(cached, direct, rtol=0, atol=1e-6)
            self.assertEqual(stats['real_slices_encoded'], 5)


class ReferenceDataTests(CPUInvariantTest):
    def test_selection_ties_empty_candidates_and_gt_schedule(self):
        rows = [dict(canal_area=0, foreground_area=0) for _ in range(4)]
        self.assertEqual(select_reference(rows), 1)
        self.assertEqual(candidate_references(rows), [1])
        rows[0].update(canal_area=10, foreground_area=100)
        rows[2].update(canal_area=10, foreground_area=9)
        self.assertEqual(select_reference(rows), 0)
        self.assertEqual(candidate_references(rows), [0])
        self.assertEqual([gt_probability(i) for i in (0, 1500, 3000, 6000)], [.5, .25, 0., 0.])

    def test_cache_budget_geometry_and_target_path(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('freqssl.case_context.verify_dataset_geometry', side_effect=fake_geometry):
            cfg = fixture(Path(tmp), 'row', gt=True)
            dataset, _, manifest = build_training_data(cfg)
            plain = dataset.base
            for name in plain.names:
                if name not in plain.labeled_names:
                    (plain.root / 'label' / name).unlink()
            from PIL import Image
            for epoch in (0, 1):
                dataset.set_epoch(epoch)
                for i in range(len(dataset)):
                    with mock.patch('PIL.Image.open', wraps=Image.open) as opened:
                        sample = dataset[i]
                    reads = [Path(c.args[0]).name for c in opened.call_args_list if Path(c.args[0]).parent.name == 'label']
                    self.assertTrue(set(reads).issubset(plain.labeled_names))
                    torch.testing.assert_close(sample['image'], plain[i]['image'], rtol=0, atol=0)
                    if sample['reference_is_target']:
                        self.assertFalse(sample['reference_gt_available'])
                    valid = sample['reference_valid'] > .999
                    sums = sample['reference_probabilities'].sum(0, keepdim=True)
                    torch.testing.assert_close(sums[valid], torch.ones_like(sums[valid]), atol=1e-5, rtol=0)
            cache = dataset.cache
            cache.checked.clear()
            name = plain.names[0]
            with (cache.root / 'probabilities' / (name + '.npy')).open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                cache.read(name)

    def test_train_cycle_gt_and_prediction_only_validation(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('freqssl.case_context.verify_dataset_geometry', side_effect=fake_geometry):
            cfg = fixture(Path(tmp), gt=True)
            dataset, sampler, _ = build_training_data(cfg)
            batch = next(iter(DataLoader(dataset, batch_sampler=sampler)))
            system = StudentTeacherSystem(cfg)
            optimizer = make_optimizer(system, cfg)
            for step in (0, 3000):
                metrics = train_step(system, optimizer, batch, cfg, torch.device('cpu'), step)
                self.assertTrue(np.isfinite(metrics['loss']))
                if step == 3000:
                    self.assertEqual(metrics['reference_gt_used'], 0)
            for teacher in system.teachers.parameters():
                self.assertIsNone(teacher.grad)
            cycle_cfg = replace(cfg, reference_gt=False, reference_cycle_weight=.1)
            cycle_data, sampler, _ = build_training_data(cycle_cfg)
            cycle_batch = next(iter(DataLoader(cycle_data, batch_sampler=sampler)))
            cycle_system = StudentTeacherSystem(cycle_cfg)
            for a in cycle_system.students[0].reference_adapters:
                a.gate.data.fill_(.1)
            result = train_step(cycle_system, make_optimizer(cycle_system, cycle_cfg), cycle_batch,
                                cycle_cfg, torch.device('cpu'), 1)
            self.assertGreater(result['cycle_samples'], 0)
            self.assertGreater(result['cycle_loss'], 0)
            validation, records = prepare_validation(cfg)
            before = state_hash(system)
            scores = evaluate_volumes(system, validation, records, 'cpu')
            self.assertEqual(scores['teacher1']['overall']['cases'], 1)
            self.assertEqual(before, state_hash(system))

    def test_cycle_soft_anchor_detaches_teacher_but_not_student(self):
        logits = torch.randn(2, 20, 5, 7, requires_grad=True)
        teacher = torch.randn_like(logits).softmax(1).detach().requires_grad_()
        loss, _ = soft_cycle_loss(logits, teacher, reference_config())
        loss.backward()
        self.assertGreater(logits.grad.abs().sum(), 0)
        self.assertIsNone(teacher.grad)

    def test_end_to_end_resume_and_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('freqssl.case_context.verify_dataset_geometry', side_effect=fake_geometry):
            cfg = fixture(Path(tmp), 'row', gt=True)
            first = run_experiment(cfg, 'cpu', stop_after=1)
            self.assertEqual(first['status'], 'paused')
            resumed = run_experiment(cfg, 'cpu', resume=Path(cfg.output_dir) / 'last.pt')
            self.assertEqual(resumed['status'], 'completed')
            other = replace(cfg, output_dir=str(Path(tmp) / 'continuous'))
            complete = run_experiment(other, 'cpu')
            self.assertEqual(resumed['final_student1_hash'], complete['final_student1_hash'])
            self.assertEqual(resumed['final_teacher1_hash'], complete['final_teacher1_hash'])
            model, restored, receipt = load_single_network(Path(cfg.output_dir) / 'last.pt', 'cpu')
            self.assertEqual(restored.reference_condition, 'row')
            self.assertTrue(receipt['reference_teacher_required'])

    def test_native_diagnostics_reproduce_evaluation_and_render_attention(self):
        from experiments.analyze_reference_conditioning import analyze
        with tempfile.TemporaryDirectory() as tmp, mock.patch('freqssl.case_context.verify_dataset_geometry', side_effect=fake_geometry), \
                mock.patch('experiments.analyze_reference_conditioning.verify_dataset_geometry', side_effect=fake_geometry):
            root = Path(tmp); cfg = fixture(root)
            model = StudentTeacherSystem(cfg)
            for adapter in model.teachers[0].reference_adapters:
                adapter.gate.data.fill_(.2)
            checkpoint = root / 'candidate.pt'
            torch.save(dict(format_version=3, config=cfg.to_dict(), model=model.state_dict(), step=2), checkpoint)
            validation, records = prepare_validation(cfg)
            expected = evaluate_volumes(model, validation, records, 'cpu')['teacher1']['overall']
            result = analyze(checkpoint, root / 'analysis', 'cpu', expected_cases=1)
            self.assertEqual(result['metrics']['full']['overall'], expected)
            self.assertTrue(result['model_state_unchanged'])
            self.assertEqual(len(result['cases'][0]['attention']), 2)
            self.assertTrue((root / 'analysis/attention_mass.png').exists())
            self.assertTrue((root / 'analysis/numbering.csv').exists())
            self.assertGreater(result['cases'][0]['conditions']['full']['cost']['attention_matmul_macs'], 0)

    def test_label_free_volume_deployment_needs_no_training_cache(self):
        from freqssl.inference import predict_volume
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg=fixture(root)
            system=StudentTeacherSystem(cfg)
            checkpoint=root/'candidate.pt'
            torch.save(dict(format_version=3,config=cfg.to_dict(),model=system.state_dict(),step=2),checkpoint)
            model,_,_=load_single_network(checkpoint,'cpu')
            shutil.rmtree(cfg.reference_cache)
            array=np.random.default_rng(42).normal(size=(3,32,32)).astype(np.float32)
            plan=dict(input_shape_zyx=list(array.shape),canonical_shape=list(array.shape),
                transpose_zyx_to_slice_ap_si=[0,1,2],flip_canonical_axes=[])
            with mock.patch('PIL.Image.open',side_effect=AssertionError('Deployment must not open PNG labels or training inputs')):
                result=predict_volume(model,array,plan,32,'cpu')
            self.assertEqual(result.shape,array.shape)
            self.assertEqual(result.dtype,np.uint8)
            self.assertEqual(model.reference_receipt['backbones_required'],2)
            self.assertGreater(model.reference_receipt['target']['attention_matmul_macs'],0)


class ReferenceProtocolTests(CPUInvariantTest):
    def test_numbering_denominators_separate_misses_from_wrong_levels(self):
        cm = np.zeros((20, 20), np.int64)
        cm[1, 1], cm[1, 2], cm[1, 0], cm[11, 13] = 40, 10, 50, 20
        result = numbering_metrics(cm)
        self.assertEqual(result['numbered_gt'], 120)
        self.assertEqual(result['tissue_correct'], 70)
        self.assertEqual(result['wrong_one'], 10)
        self.assertEqual(result['wrong_multiple'], 20)
        self.assertEqual(result['wrong_level_percent'], 25.)
        self.assertAlmostEqual(result['conditional_wrong_level_percent'], 100*30/70)
        self.assertIsNone(numbering_metrics(np.zeros((20, 20)))['conditional_wrong_level_percent'])

    def test_selection_confirmation_and_cycle_are_separate_gates(self):
        from experiments.run_reference_conditioning import ARMS, choose_candidate, confirmation_arms, confirmation_result, cycle_eligible, cycle_screen_passed
        def row(dice, union=.8, wrong=5., macs=100):
            return dict(overall=dict(dice=dice, dice_union_present=union), pipeline_macs=macs,
                        numbering=dict(conditional_wrong_level_percent=wrong))
        rows = {'A':row(.8), **{a:row(.802) for a in ARMS}}
        self.assertIsNone(choose_candidate(rows))
        rows['KV'] = row(.805, wrong=4.)
        rows['KV-GT'] = row(.806, union=.79)
        self.assertEqual(choose_candidate(rows), 'KV')
        self.assertEqual(confirmation_arms('KV-GT'), ['P1','KV-pos','KV','KV-GT'])
        runs = {str(s):rows.copy() for s in (42,43,44)}
        self.assertTrue(confirmation_result(runs,'KV')['passed'])
        self.assertTrue(cycle_eligible(runs))
        rows['KV-cycle'] = row(.807, wrong=4.)
        self.assertTrue(cycle_screen_passed(rows))
        rows['KV-cycle'] = row(.807, wrong=6.)
        self.assertFalse(cycle_screen_passed(rows))

    def test_prepare_only_freezes_configs_and_never_launches(self):
        from experiments.run_reference_conditioning import prepare, ROOT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); geometry = root / 'geometry.json'; geometry.write_text('{}')
            base = __import__('freqssl.config', fromlist=['load_config']).load_config(ROOT / 'configs/v3/case_reference.yaml')
            base = replace(base, context_geometry_manifest=str(geometry))
            config = root / 'config.yaml'; config.write_text(json.dumps(base.to_dict()))
            baseline = dict(labeled_files=['l'], unlabeled_files=['u'], validation_cases=['v'],
                checkpoint='/fixture/last.pt', checkpoint_sha256='fixture')
            args = SimpleNamespace(output_dir=root / 'prepared', config=config, baselines=None,
                legacy_snapshot=ROOT, device='cuda:0', screen_only=False)
            with mock.patch('experiments.run_reference_conditioning.validate_baseline', return_value=baseline), \
                 mock.patch('experiments.run_reference_conditioning.verify_legacy_baseline', return_value={'exact':True}), \
                 mock.patch('experiments.run_reference_conditioning.build_training_data', return_value=(None,None,baseline)), \
                 mock.patch('experiments.run_reference_conditioning.architecture_profiles', return_value={}), \
                 mock.patch('experiments.run_reference_conditioning.run_experiment') as train, \
                 mock.patch('experiments.run_reference_conditioning.subprocess.Popen') as launch:
                command = prepare(args)
            train.assert_not_called(); launch.assert_not_called()
            self.assertEqual(json.loads((args.output_dir/'status.json').read_text())['status'], 'prepared')
            for seed in (42,43,44):
                files = list((args.output_dir/'configs'/f'seed{seed}').glob('*.json'))
                self.assertEqual(len(files), 7)
                for path in files:
                    cfg = ExperimentConfig(**json.loads(path.read_text())).validate()
                    self.assertEqual(cfg.max_steps, 6000)
            self.assertEqual(command[-1], '--worker')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; GPU gradient/cache checks required at formal preflight')
    def test_cuda_nonzero_gate_and_cache(self):
        cfg = reference_config()
        model = StudentTeacherSystem(cfg).students[0].cuda().train()
        ref = {k:v.cuda() for k,v in reference_batch(2).items()}
        images = torch.randn(2,3,32,32,device='cuda')
        model(images, reference=ref).square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        for adapter in model.reference_adapters:
            adapter.gate.data.fill_(.2)
        model.eval()
        volume = images[0].detach().cpu()[:,None]
        probabilities = torch.randn(3,20,32,32).softmax(1)
        values = torch.cat([v for _,v in iter_reference_logits(model,3,lambda i:volume[i],lambda i:probabilities[i], 'cuda',reference_index=1)])
        self.assertEqual(values.shape,(3,20,32,32))
        self.assertTrue(torch.isfinite(values).all())


if __name__ == '__main__':
    unittest.main()
