"""Checkpoint continuation, frozen fields and optimizer surgery regressions."""
import importlib
import json
import random
import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer.composition import OpticalCorrection
from scene.deformation import Deformation
from scene.gaussian_model import GaussianModel
from utils.checkpoint_utils import checkpoint, load_field, restore_rng_state
from metrics import evaluate, performance_summary, phase_training_seconds, update_performance


class ImportTests(unittest.TestCase):
    def test_render_import_does_not_require_training_entry_point(self):
        with patch.dict(sys.modules, {'train': None}):
            renderer = importlib.import_module('render')
            self.assertIs(renderer.load_field, load_field)


class PerformanceReportingTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(prefix='vigor-performance-')
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)

    def logs(self,phase,records):
        path=self.root/phase/'training.jsonl'
        path.parent.mkdir(exist_ok=True)
        path.write_text(''.join(json.dumps(record)+'\n' for record in records))
        return path

    def test_legacy_elapsed_records_are_not_summed_per_iteration(self):
        self.logs('interface',[dict(iteration=50,elapsed_seconds=3600.),
                               dict(iteration=100,elapsed_seconds=7200.)])
        self.logs('reflection',[dict(iteration=100,elapsed_seconds=300.)])
        (self.root/'render_stats.json').write_text(json.dumps({'test':dict(frames=29,mean_ms=25.)}))
        summary=performance_summary(self.root,'ours_100','test',29,sole_method=True)
        self.assertEqual(summary['FPS'],40.)
        self.assertEqual(summary['Training Time (seconds)'],7500.)
        self.assertEqual(summary['Training Time'],'2 hours and 5 mins')
        self.assertEqual(summary['Training Time by Phase (seconds)'],dict(interface=7200.,reflection=300.))

    def test_resumed_sessions_accumulate_even_when_elapsed_time_increases(self):
        path=self.logs('reflection',[
            dict(iteration=10,elapsed_seconds=50.,session_id='first'),
            dict(iteration=20,elapsed_seconds=100.,session_id='first'),
            dict(iteration=30,elapsed_seconds=150.,session_id='resume'),
            dict(iteration=40,elapsed_seconds=200.,session_id='resume')])
        self.assertEqual(phase_training_seconds(path),300.)
        self.assertEqual(phase_training_seconds(path,30),250.)
        legacy=self.logs('interface',[dict(iteration=20,elapsed_seconds=100.),
            dict(iteration=30,elapsed_seconds=10.),dict(iteration=40,elapsed_seconds=20.)])
        self.assertEqual(phase_training_seconds(legacy),120.)

    def test_method_specific_fps_and_missing_measurements(self):
        stats={'test':dict(method='direct_100',frames=29,mean_ms=5.),
               'methods':{'test':{'ours_100':dict(frames=29,mean_ms=25.)}}}
        (self.root/'render_stats.json').write_text(json.dumps(stats))
        summary=performance_summary(self.root,'ours_100','test',29)
        self.assertEqual(summary['FPS'],40.)
        self.assertIsNone(summary['Training Time'])
        self.assertIsNone(summary['Training Time (seconds)'])
        self.assertIsNone(performance_summary(self.root,'ours_200','test',29)['FPS'])
        self.assertIsNone(performance_summary(self.root,'ours_100','test',1)['FPS'])
        self.assertIsNone(performance_summary(self.root,'ours_100','train',29)['FPS'])

    def test_direct_results_only_count_interface_training(self):
        self.logs('interface',[dict(iteration=100,elapsed_seconds=120.)])
        self.logs('reflection',[dict(iteration=100,elapsed_seconds=900.)])
        summary=performance_summary(self.root,'direct_100','test',29)
        self.assertEqual(summary['Training Time'],'0 hours and 2 mins')
        self.assertEqual(summary['Training Time (seconds)'],120.)

    def test_metrics_export_and_performance_refresh_preserve_image_scores(self):
        from PIL import Image
        method=self.root/'test/ours_100'
        for sub in ('renders','gt','transparent_masks','valid_masks'):
            folder=method/sub
            folder.mkdir(parents=True)
            array=np.full((12,12,3),128,dtype=np.uint8) if sub in ('renders','gt') else np.full((12,12),255,dtype=np.uint8)
            Image.fromarray(array).save(folder/'frame.png')
        self.logs('interface',[dict(iteration=100,elapsed_seconds=7200.)])
        self.logs('reflection',[dict(iteration=100,elapsed_seconds=300.)])
        (self.root/'render_stats.json').write_text(json.dumps({'test':dict(method='ours_100',frames=1,mean_ms=20.)}))
        evaluate([self.root],device='cpu',skip_lpips=True)
        original=json.loads((self.root/'results.json').read_text())['ours_100']
        self.assertEqual(original['FPS'],50.)
        self.assertEqual(original['Training Time'],'2 hours and 5 mins')
        (self.root/'render_stats.json').write_text(json.dumps({'test':dict(method='ours_100',frames=1,mean_ms=25.)}))
        update_performance([self.root])
        updated=json.loads((self.root/'results.json').read_text())['ours_100']
        self.assertEqual(updated['FPS'],40.)
        for key,value in original.items():
            if key!='FPS': self.assertEqual(updated[key],value)
        per_view=json.loads((self.root/'per_view.json').read_text())['ours_100']['frame.png']
        self.assertNotIn('FPS',per_view)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        parser = ArgumentParser()
        ModelParams(parser)
        PipelineParams(parser)
        OptimizationParams(parser)
        self.args = parser.parse_args([])
        self.args.sh_degree = 1
        self.args.deform_depth, self.args.deform_width = 2, 16
        self.args.deform_xyz_freqs, self.args.deform_time_freqs = 2, 2
        self.args.max_interface_points = self.args.max_reflection_points = 20
        self.folder = tempfile.TemporaryDirectory(prefix='vigor-checkpoint-')
        self.addCleanup(self.folder.cleanup)

    def field(self, kind):
        torch.manual_seed(19)
        field = GaussianModel(self.args.sh_degree, self.args, field=kind)
        field.create_from_points(torch.rand(16, 3, device='cuda')*.2,
                                 torch.rand(16, 3, device='cuda'), 1.)
        field.training_setup(self.args)
        deformation = Deformation(field.SCALE_DIM, self.args).cuda()
        deformation.optimizer = torch.optim.Adam(deformation.parameters(), lr=self.args.deform_lr)
        optical = OpticalCorrection(self.args.optical_bound).cuda() if kind == 'reflection' else None
        optimizer = torch.optim.Adam(optical.parameters(), lr=self.args.optical_lr) if optical is not None else None
        return field, deformation, optical, optimizer

    @staticmethod
    def step(field, deformation, optical=None, optical_optimizer=None):
        field.optimizer.zero_grad(set_to_none=True)
        deformation.optimizer.zero_grad(set_to_none=True)
        loss = sum(getattr(field, '_'+name).square().mean() for name in field.PARAMS)
        loss = loss + sum(value.square().mean() for value in deformation(field, .37))
        if optical is not None:
            optical_optimizer.zero_grad(set_to_none=True)
            uv = field.get_xyz[:4, :2].detach()
            loss = loss + (optical(uv, .37, uv.new_ones(4))-.01).square().mean()
        loss.backward()
        field.optimizer.step()
        deformation.optimizer.step()
        if optical_optimizer is not None:
            optical_optimizer.step()

    def assert_nested_close(self, actual, expected):
        if torch.is_tensor(actual):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_nested_close(actual[key], expected[key])
        elif isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_nested_close(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_resume_matches_the_next_uninterrupted_optimizer_step(self):
        for kind in ('interface', 'reflection'):
            with self.subTest(field=kind):
                field, deformation, optical, optimizer = self.field(kind)
                self.step(field, deformation, optical, optimizer)
                path = Path(self.folder.name)/(kind+'.pth')
                checkpoint(path, field, deformation, 7, kind, optical, optimizer)
                expected_rng = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, device='cuda'))
                self.step(field, deformation, optical, optimizer)

                restored, network, state = load_field(path, self.args, kind, 1., torch.zeros(3), training=True)
                correction = OpticalCorrection(self.args.optical_bound).cuda() if optical is not None else None
                correction_optimizer = None
                if correction is not None:
                    correction.load_state_dict(state['optical'])
                    correction_optimizer = torch.optim.Adam(correction.parameters(), lr=self.args.optical_lr)
                    correction_optimizer.load_state_dict(state['optical_optimizer'])
                restore_rng_state(state)
                actual_rng = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, device='cuda'))
                self.assert_nested_close(actual_rng, expected_rng)
                self.assertEqual(state['version'], 1)
                self.assertEqual(state['iteration'], 7)
                self.step(restored, network, correction, correction_optimizer)
                for name in field.PARAMS:
                    self.assert_nested_close(getattr(restored, '_'+name), getattr(field, '_'+name))
                self.assert_nested_close(network.state_dict(), deformation.state_dict())
                self.assert_nested_close(restored.optimizer.state_dict(), field.optimizer.state_dict())
                self.assert_nested_close(network.optimizer.state_dict(), deformation.optimizer.state_dict())
                if correction is not None:
                    self.assert_nested_close(correction.state_dict(), optical.state_dict())
                    self.assert_nested_close(correction_optimizer.state_dict(), optimizer.state_dict())

    def test_inference_freezes_both_field_and_deformation(self):
        field, deformation, _, _ = self.field('interface')
        self.step(field, deformation)
        path = Path(self.folder.name)/'interface.pth'
        checkpoint(path, field, deformation, 1, 'interface')
        frozen, network, _ = load_field(path, self.args, 'interface', 1., torch.zeros(3))
        self.assertIsNone(frozen.optimizer)
        self.assertIsNone(network.optimizer)
        self.assertFalse(network.training)
        self.assertFalse(any(getattr(frozen, '_'+name).requires_grad for name in frozen.PARAMS))
        self.assertFalse(any(p.requires_grad for p in network.parameters()))
        self.assertFalse(any(value.requires_grad for value in network(frozen, .5)))

    def test_resume_rejects_a_point_cap_smaller_than_checkpoint(self):
        field, deformation, _, _ = self.field('interface')
        path = Path(self.folder.name)/'interface.pth'
        checkpoint(path, field, deformation, 1, 'interface')
        self.args.max_interface_points = 8
        with self.assertRaisesRegex(ValueError, 'point cap'):
            load_field(path, self.args, 'interface', 1., torch.zeros(3), training=True)

    def test_growth_cap_and_optimizer_state_survive_clone_and_split(self):
        for kind in ('interface', 'reflection'):
            with self.subTest(field=kind):
                field, deformation, optical, optimizer = self.field(kind)
                self.step(field, deformation, optical, optimizer)
                with torch.no_grad():
                    field._scaling[:8].fill_(np.log(.001))
                    field._scaling[8:].fill_(np.log(.1))
                    field.xyz_gradient_accum.fill_(1.)
                    # Highest scores alternate small and large Gaussians.
                    field.xyz_gradient_accum[[0, 8, 1, 9], 0] = torch.tensor([5., 4., 3., 2.], device='cuda')
                    field.denom.fill_(1.)
                field.densify(.5, 1.)
                self.assertEqual(field.num_points, 20)
                for group in field.optimizer.param_groups:
                    parameter = group['params'][0]
                    self.assertEqual(parameter.shape[0], 20)
                    self.assertEqual(field.optimizer.state[parameter]['exp_avg'].shape, parameter.shape)
                self.step(field, deformation, optical, optimizer)
                self.assertTrue(torch.isfinite(field.get_xyz).all())


if __name__ == '__main__':
    unittest.main()
