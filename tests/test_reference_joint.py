"""Synthetic invariants for neighbour (i-1, i+1) and joint KV-pos banks.

These check wiring, pairing and inference equivalence only. They say nothing
about whether the joint arm segments better; that needs the GPU runs.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from freqssl.config import ExperimentConfig, load_config
from freqssl.data import build_training_data, _case_id
from freqssl.engine import make_optimizer, train_step, run_experiment, state_hash
from freqssl.evaluation import prepare_validation, evaluate_volumes
from freqssl.inference import load_single_network
from freqssl.models import StudentTeacherSystem
from freqssl.reference_conditioning import si_coordinates
from freqssl.reference_inference import iter_reference_logits
from test_freqssl import CPUInvariantTest
from test_case_context import fake_geometry
from test_reference_conditioning import reference_config, reference_batch, fixture

ROOT = Path(__file__).resolve().parents[1]


def neighbor_config(sources='case_neighbors', **kwargs):
    return replace(reference_config(), reference_condition='position', reference_sources=sources, **kwargs).validate()


def neighbor_reference(b=2, shape=(32, 32), present=None):
    reference = reference_batch(b, shape)
    valid = torch.ones(b, 2, *shape)
    if present is not None:
        valid = valid * torch.as_tensor(present, dtype=torch.float32)[..., None, None]
    reference.update(neighbor_image=torch.randn(b, 2, *shape),
                     neighbor_probabilities=torch.randn(b, 2, 20, *shape).softmax(2),
                     neighbor_valid=valid)
    return reference


def open_gates(model, value=.3):
    for adapter in [*model.reference_adapters, *model.neighbor_adapters]:
        adapter.gate.data.fill_(value)


class NeighborConfigTests(CPUInvariantTest):
    def test_defaults_and_rejected_combinations(self):
        self.assertEqual(reference_config().reference_sources, 'case')
        for sources in ('neighbors', 'case_neighbors'):
            self.assertEqual(neighbor_config(sources).reference_sources, sources)
        with self.assertRaises(ValueError):
            replace(reference_config(), reference_sources='neighbors').validate()  # image-key KV
        with self.assertRaises(ValueError):
            replace(reference_config(), reference_sources='all').validate()
        with self.assertRaises(ValueError):
            neighbor_config(reference_gt=True)
        with self.assertRaises(ValueError):
            replace(reference_config(), reference_condition='none', reference_teacher='',
                    reference_sources='neighbors').validate()

    def test_repository_configs_parse_and_pair(self):
        directory = ROOT / 'configs/v3/reference_joint'
        for seed in (42, 43, 44):
            arms = {path.stem: load_config(path) for path in sorted((directory / f'seed{seed}').glob('*.yaml'))}
            self.assertEqual(set(arms), {'KV-pos', 'NB-pos', 'JOINT-pos'})
            self.assertEqual([arms[a].reference_sources for a in ('KV-pos', 'NB-pos', 'JOINT-pos')],
                             ['case', 'neighbors', 'case_neighbors'])
            ignore = {'output_dir', 'reference_sources'}
            values = [{k: v for k, v in cfg.to_dict().items() if k not in ignore} for cfg in arms.values()]
            self.assertTrue(all(v == values[0] for v in values), seed)
            self.assertEqual(arms['JOINT-pos'].seed, seed)
            self.assertEqual((arms['JOINT-pos'].max_steps, arms['JOINT-pos'].reference_condition), (6000, 'position'))
            self.assertIn(f'seed{seed}', arms['JOINT-pos'].reference_cache)
            self.assertEqual(len({cfg.output_dir for cfg in arms.values()}), 3)
        # The summary plan reads exactly the directories these configs train into.
        import yaml
        from freqssl.paths import expand
        plan = expand(yaml.safe_load((ROOT / 'configs/plans/reference_joint_20261127.yaml').read_text()))
        for seed in (42, 43, 44):
            for arm in ('KV-pos', 'NB-pos', 'JOINT-pos'):
                cfg = load_config(directory / f'seed{seed}/{arm}.yaml')
                self.assertEqual(plan['arms'][arm][seed][0], cfg.output_dir)
        # The case-only arm must equal the original KV-pos protocol except paths.
        original = load_config(ROOT / 'configs/v3/reference_conditioning/KV-pos.yaml').to_dict()
        rerun = load_config(directory / 'seed42/KV-pos.yaml').to_dict()
        paths = {'output_dir', 'reference_cache', 'max_seconds'}
        self.assertEqual({k: v for k, v in original.items() if k not in paths},
                         {k: v for k, v in rerun.items() if k not in paths})


class NeighborModelTests(CPUInvariantTest):
    def test_initialization_pairs_arms_and_starts_exactly_at_a(self):
        plain = StudentTeacherSystem(replace(reference_config(), reference_condition='none',
                                             reference_teacher='')).students[0]
        state = torch.get_rng_state().clone()
        models = {s: StudentTeacherSystem(neighbor_config(s)).students[0] for s in ('case', 'neighbors', 'case_neighbors')}
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        joint = models['case_neighbors'].state_dict()
        for source in ('case', 'neighbors'):
            for key, value in models[source].state_dict().items():
                torch.testing.assert_close(joint[key], value, rtol=0, atol=0)
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(joint[key], value, rtol=0, atol=0)
        self.assertFalse(torch.equal(joint['mask_encoder.layers.0.0.weight'],
                                     joint['neighbor_mask_encoder.layers.0.0.weight']))
        self.assertFalse(any(k.startswith('neighbor') for k in models['case'].state_dict()))
        self.assertFalse(any(k.startswith(('mask_encoder', 'reference_adapters')) for k in models['neighbors'].state_dict()))
        for module in (plain, *models.values()):
            for fusion in module.context_fusion:
                fusion.gamma.data.fill_(.3)
        images, reference = torch.randn(2, 3, 32, 32), neighbor_reference()
        expected = plain(images)
        for source, model in models.items():
            torch.testing.assert_close(model(images, reference=reference), expected, rtol=0, atol=0, msg=source)

    def test_gradients_reach_gates_then_neighbor_values_but_never_keys(self):
        model = StudentTeacherSystem(neighbor_config()).students[0]
        images, reference = torch.randn(2, 3, 32, 32), neighbor_reference()
        model(images, reference=reference).square().mean().backward()
        self.assertTrue(all(m.gate.grad.abs() > 0 for m in model.neighbor_adapters))
        model.zero_grad()
        open_gates(model, .1)
        model(images, reference=reference).square().mean().backward()
        self.assertGreater(model.neighbor_mask_encoder.layers[0][0].weight.grad.abs().sum(), 0)
        self.assertTrue(all(m.query.weight.grad.abs().sum() > 0 for m in model.neighbor_adapters))
        self.assertTrue(all(m.key.weight.grad is None for m in model.neighbor_adapters))

    def test_absent_slots_patients_and_slot_order(self):
        model = StudentTeacherSystem(neighbor_config('neighbors')).students[0]
        open_gates(model, .5)
        images = torch.randn(2, 3, 32, 32)
        reference = neighbor_reference(present=[[1, 0], [1, 1]])
        # Inspect the stride-8 injection itself, before decoder convolutions.
        with mock.patch.object(model, 'decode_features', side_effect=lambda skips: skips[3]), \
                mock.patch.object(model, 'project', side_effect=lambda features: features):
            first = model(images, reference=reference)
            changed = {k: v.clone() for k, v in reference.items()}
            changed['neighbor_image'][0, 1] += 50
            changed['neighbor_probabilities'][0, 1] = changed['neighbor_probabilities'][0, 1].roll(3, 0)
            # An absent slot is never read.
            torch.testing.assert_close(model(images, reference=changed), first, rtol=0, atol=0)
            changed['neighbor_image'][1, 0] += 50
            second = model(images, reference=changed)
            # Another patient's bank never enters this sample's softmax.
            torch.testing.assert_close(second[0], first[0], rtol=0, atol=0)
            self.assertFalse(torch.equal(second[1], first[1]))
            # Position keys are equal for both slots, so slot order cannot matter:
            # the bank is the attention-weighted mean of the two neighbours.
            swapped = {k: v.clone() for k, v in reference.items()}
            for key in ('neighbor_image', 'neighbor_probabilities', 'neighbor_valid'):
                swapped[key][1] = swapped[key][1].flip(0)
            torch.testing.assert_close(model(images, reference=swapped)[1], first[1], rtol=0, atol=1e-6)
            # With every slot absent the residual is exactly zero, as if the gate were closed.
            empty = {k: v.clone() for k, v in reference.items()}
            empty['neighbor_valid'].zero_()
            opened = model(images, reference=empty)
            open_gates(model, 0.)
            torch.testing.assert_close(opened, model(images, reference=reference), rtol=0, atol=0)

    def test_cutmix_routes_neighbor_banks_by_donor(self):
        model = StudentTeacherSystem(neighbor_config()).students[0]
        open_gates(model, .5)
        reference = neighbor_reference()
        bank = model.prepare_reference(reference['image'], reference['probabilities'], reference['si'], reference['valid'])
        neighbors = model.prepare_neighbors(reference)
        encoded = model.encode(torch.randn(6, 1, 32, 32))
        triplets = [v.reshape(2, 3, *v.shape[1:]) for v in encoded]
        full = torch.zeros(2, 1, 32, 32); full[0] = 1
        donors = torch.tensor([1, 0])
        with mock.patch.object(model, 'decode_features', side_effect=lambda skips: skips[3]):
            first = model.decode_reference(triplets, bank, reference['si'], reference['valid'], donors, full, neighbors)
            changed = [{k: v.clone() for k, v in level.items()} for level in neighbors]
            for level in changed:
                level['value'][0] += 100
            second = model.decode_reference(triplets, bank, reference['si'], reference['valid'], donors, full, changed)
        # Sample 0 is wholly its donor; sample 1 keeps its own bank.
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_cached_volume_inference_matches_direct_forward(self):
        for sources in ('neighbors', 'case_neighbors'):
            model = StudentTeacherSystem(neighbor_config(sources)).students[0].eval()
            open_gates(model, .3)
            count, r = 11, 4  # two output microbatches and a neighbour chunk boundary
            images = torch.randn(count, 1, 32, 32)
            p = torch.randn(count, 20, 32, 32).softmax(1)
            triplets = torch.cat([images[[max(0, i-1), i, min(count-1, i+1)]].transpose(0, 1) for i in range(count)])
            present = torch.tensor([[i > 0, i < count - 1] for i in range(count)], dtype=torch.float32)
            take = lambda x, d: torch.stack([x[min(count-1, max(0, i+d))] for i in range(count)])
            reference = dict(image=images[r:r+1].expand(count, -1, -1, -1), probabilities=p[r:r+1].expand(count, -1, -1, -1),
                si=si_coordinates(count, (32, 32)), valid=torch.ones(count, 1, 32, 32),
                neighbor_image=torch.cat((take(images, -1), take(images, 1)), 1) * present[..., None, None],
                neighbor_probabilities=torch.stack((take(p, -1), take(p, 1)), 1) * present[..., None, None, None],
                neighbor_valid=present[..., None, None].expand(-1, -1, 32, 32))
            with torch.no_grad():
                direct = model(triplets, reference=reference)
                stats = {}
                cached = torch.cat([v for _, v in iter_reference_logits(model, count, lambda i: images[i],
                    lambda i: p[i], 'cpu', 3, reference_index=r, stats=stats)])
                disabled = torch.cat([v for _, v in iter_reference_logits(model, count, lambda i: images[i],
                    lambda i: p[i], 'cpu', 3, reference_index=r, disable=True)])
                plain = model.context_features([v.reshape(count, 3, *v.shape[1:]) for v in
                                                model.encode(triplets.reshape(-1, 1, 32, 32))])
            torch.testing.assert_close(cached, direct, rtol=0, atol=1e-5, msg=sources)
            torch.testing.assert_close(disabled, model.project(plain), rtol=0, atol=1e-5, msg=sources)
            self.assertFalse(torch.allclose(cached, disabled, atol=1e-4))
            self.assertEqual(stats['real_slices_encoded'], count)
            self.assertEqual(stats['neighbor_values_encoded'], count)


class NeighborPipelineTests(CPUInvariantTest):
    def test_dataset_keeps_case_fields_and_adds_neighbors_in_target_affine(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('freqssl.case_context.verify_dataset_geometry', side_effect=fake_geometry):
            cfg = fixture(Path(tmp), 'position')
            case, _, _ = build_training_data(cfg)
            joint, _, manifest = build_training_data(replace(cfg, reference_sources='case_neighbors'))
            self.assertEqual(manifest['reference_conditioning']['condition'], 'position')
            for epoch in (0, 1):
                case.set_epoch(epoch); joint.set_epoch(epoch)
                for i in range(len(joint)):
                    a, b = case[i], joint[i]
                    for key, value in a.items():
                        if torch.is_tensor(value):
                            torch.testing.assert_close(b[key], value, rtol=0, atol=0, msg=key)
                        else:
                            self.assertEqual(b[key], value, key)
                    names = joint.cache.groups[_case_id(joint.base.names[i])]
                    target = names.index(joint.base.names[i])
                    expected = [0 <= target + d < len(names) for d in (-1, 1)]
                    self.assertEqual(b['reference_neighbor_present'].tolist(), expected)
                    for slot, present in enumerate(expected):
                        if present:
                            torch.testing.assert_close(b['reference_neighbor_valid'][slot], b['reference_valid'][0], rtol=0, atol=0)
                            sums = b['reference_neighbor_probabilities'][slot].sum(0)
                            inside = b['reference_valid'][0] > .999
                            torch.testing.assert_close(sums[inside], torch.ones_like(sums[inside]), atol=1e-5, rtol=0)
                        else:
                            self.assertEqual(b['reference_neighbor_valid'][slot].count_nonzero(), 0)
                            self.assertEqual(b['reference_neighbor_image'][slot].count_nonzero(), 0)
                    # A neighbour probability map is the warped cached map of that slice.
                    if expected[1]:
                        source = joint.cache.read(names[target + 1])
                        _, _, warped = joint.base._geometry(torch.zeros(1, 32, 32), None, i,
                            torch.cat((source, torch.ones(1, 32, 32))), 'bilinear')
                        torch.testing.assert_close(b['reference_neighbor_probabilities'][1],
                                                   warped[:20] / warped[20:].clamp_min(1e-8), rtol=0, atol=0)

    def test_training_steps_evaluation_resume_and_deployment(self):
        from freqssl.inference import predict_volume
        import shutil
        for sources in ('neighbors', 'case_neighbors'):
            with tempfile.TemporaryDirectory() as tmp, mock.patch('freqssl.case_context.verify_dataset_geometry', side_effect=fake_geometry):
                cfg = replace(fixture(Path(tmp), 'position'), reference_sources=sources, cutmix_probability=1.).validate()
                dataset, sampler, _ = build_training_data(cfg)
                batch = next(iter(DataLoader(dataset, batch_sampler=sampler)))
                system = StudentTeacherSystem(cfg)
                open_gates(system.students[0], .2)
                metrics = train_step(system, make_optimizer(system, cfg), batch, cfg, torch.device('cpu'), 1)
                self.assertTrue(np.isfinite(metrics['loss']))
                self.assertEqual(len(metrics['reference_neighbor_gates']), 2)
                self.assertEqual(len(metrics['reference_gates']), 2 if sources == 'case_neighbors' else 0)
                self.assertTrue(all(p.grad is None for p in system.teachers.parameters()))
                validation, records = prepare_validation(cfg)
                before = state_hash(system)
                scores = evaluate_volumes(system, validation, records, 'cpu')
                self.assertEqual(scores['teacher1']['overall']['cases'], 1)
                self.assertEqual(before, state_hash(system))
                first = run_experiment(cfg, 'cpu', stop_after=1)
                self.assertEqual(first['status'], 'paused')
                resumed = run_experiment(cfg, 'cpu', resume=Path(cfg.output_dir) / 'last.pt')
                complete = run_experiment(replace(cfg, output_dir=str(Path(tmp) / 'continuous')), 'cpu')
                self.assertEqual(resumed['final_student1_hash'], complete['final_student1_hash'])
                model, restored, receipt = load_single_network(Path(cfg.output_dir) / 'last.pt', 'cpu')
                self.assertEqual((restored.reference_sources, receipt['reference_sources']), (sources, sources))
                self.assertEqual(model.backbone.uses_neighbor_reference, True)
                shutil.rmtree(cfg.reference_cache)
                array = np.random.default_rng(42).normal(size=(3, 32, 32)).astype(np.float32)
                plan = dict(input_shape_zyx=list(array.shape), canonical_shape=list(array.shape),
                            transpose_zyx_to_slice_ap_si=[0, 1, 2], flip_canonical_axes=[])
                with mock.patch('PIL.Image.open', side_effect=AssertionError('Deployment must not open PNGs')):
                    result = predict_volume(model, array, plan, 32, 'cpu')
                self.assertEqual(result.shape, array.shape)
                self.assertEqual(model.reference_receipt['target']['neighbor_values_encoded'], 3)


class JointScriptTests(CPUInvariantTest):
    def test_cache_builder_builds_then_verifies_and_rejects_wrong_teacher(self):
        import json
        from experiments.build_reference_cache import main as build
        with tempfile.TemporaryDirectory() as tmp, mock.patch('freqssl.case_context.verify_dataset_geometry', side_effect=fake_geometry):
            cfg = replace(fixture(Path(tmp), 'position'), reference_sources='case_neighbors',
                          reference_cache=str(Path(tmp) / 'new_cache')).validate()
            config = Path(tmp) / 'joint.yaml'; config.write_text(json.dumps(cfg.to_dict()))
            build(['--config', str(config), '--device', 'cpu'])
            receipt = json.loads((Path(tmp) / 'new_cache/receipt.json').read_text())
            self.assertEqual([receipt['splits'][s]['action'] for s in ('train', 'valid')], ['built', 'built'])
            build(['--config', str(config), '--device', 'cpu'])
            receipt = json.loads((Path(tmp) / 'new_cache/receipt.json').read_text())
            self.assertEqual([receipt['splits'][s]['action'] for s in ('train', 'valid')], ['verified', 'verified'])
            # The built cache is what training reads.
            dataset, _, manifest = build_training_data(cfg)
            self.assertEqual(manifest['reference_conditioning']['teacher_sha256'], receipt['teacher_sha256'])
            config.write_text(json.dumps(replace(cfg, reference_teacher_sha256='0' * 64).to_dict()))
            with self.assertRaises(ValueError):
                build(['--config', str(config), '--device', 'cpu'])
            config.write_text(json.dumps(replace(cfg, seed=cfg.seed + 1).to_dict()))
            with self.assertRaises(ValueError):
                build(['--config', str(config), '--device', 'cpu'])

    def test_summary_pairs_cases_and_applies_fixed_gates(self):
        import json
        import yaml
        from experiments.summarize_reference_joint import main as summarize
        gains = {'A': 0., 'KV-pos': .4, 'NB-pos': .2, 'JOINT-pos': .6}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(step=6000, seeds=[42, 43, 44], output=str(root / 'summary'), arms={},
                        decision=dict(joint_vs_best_single_dice_pp=.1, joint_vs_a_dice_pp=.3, min_positive_seeds=2))
            rng = np.random.default_rng(0)
            base = rng.uniform(.7, .9, size=(3, 21))
            for arm, gain in gains.items():
                plan['arms'][arm] = {}
                for i, seed in enumerate((42, 43, 44)):
                    run = root / arm / str(seed); run.mkdir(parents=True)
                    cm = np.zeros((20, 20), int); cm[1, 1], cm[1, 2] = 90, 10
                    per_case = [dict(case=f'c{c}', dice=base[i, c] + gain / 100, dice_union_present=base[i, c] - .01)
                                for c in range(21)]
                    overall = dict(dice=float(np.mean([r['dice'] for r in per_case])),
                                   dice_union_present=float(np.mean([r['dice_union_present'] for r in per_case])), cases=21)
                    cases = [dict(case=f'c{c}', anatomy=dict(confusion_gt_rows_pred_columns=cm.tolist())) for c in range(21)]
                    (run / 'valid_step_6000.json').write_text(json.dumps(dict(teacher1=dict(overall=overall, per_case=per_case, cases=cases))))
                    (run / 'result.json').write_text(json.dumps(dict(status='completed', step=6000)))
                    (run / 'config.json').write_text(json.dumps(dict(seed=seed)))
                    plan['arms'][arm][seed] = [str(root / 'missing'), str(run)]
            (root / 'plan.yaml').write_text(yaml.safe_dump(plan))
            summarize(['--plan', str(root / 'plan.yaml')])
            summary = json.loads((root / 'summary/summary.json').read_text())
            decision = summary['decision']
            self.assertEqual((decision['status'], decision['best_single']), ('complete', 'KV-pos'))
            self.assertTrue(decision['joint_beats_best_single'])
            self.assertTrue(decision['joint_beats_a'])
            row = summary['comparisons']['JOINT-pos_minus_KV-pos']['dice']
            self.assertAlmostEqual(row['mean_pp'], .2, places=6)
            self.assertEqual((row['positive_seeds'], row['cases']), (3, 21))
            self.assertTrue((root / 'summary/per_case.csv').exists())
            # A missing seed makes the decision partial instead of guessing.
            for arm in gains:
                plan['arms'][arm][44] = [str(root / 'missing')]
            (root / 'plan.yaml').write_text(yaml.safe_dump(plan))
            summarize(['--plan', str(root / 'plan.yaml')])
            self.assertEqual(json.loads((root / 'summary/summary.json').read_text())['decision']['status'], 'partial')


class NeighborCudaTests(CPUInvariantTest):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; run this on the training GPU before formal runs')
    def test_cuda_deterministic_backward_and_volume_inference(self):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            model = StudentTeacherSystem(neighbor_config()).students[0].cuda().train()
            open_gates(model, .2)
            reference = {k: v.cuda() for k, v in neighbor_reference().items()}
            images = torch.randn(2, 3, 32, 32, device='cuda')
            model(images, reference=reference).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertGreater(model.neighbor_mask_encoder.layers[0][0].weight.grad.abs().sum(), 0)
            model.eval()
            volume = torch.randn(3, 1, 32, 32)
            probabilities = torch.randn(3, 20, 32, 32).softmax(1)
            values = torch.cat([v for _, v in iter_reference_logits(model, 3, lambda i: volume[i],
                lambda i: probabilities[i], 'cuda', reference_index=1)])
            self.assertEqual(values.shape, (3, 20, 32, 32))
            self.assertTrue(torch.isfinite(values).all())
        finally:
            torch.use_deterministic_algorithms(previous)
