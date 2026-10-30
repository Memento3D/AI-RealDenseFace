"""Projection math, source binding and failure handling for the personal hair review."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import im2_diagnostics as diag
import run_im2_review as suite
from run_im2_sample import load_overrides
from im2_support import PIN, sha256


class ProjectionTests(unittest.TestCase):
    def test_known_camera_and_behind_camera(self):
        camera = np.array([[100, 0, 20, 0], [0, 100, 30, 0], [0, 0, 1, 0]])
        uv, z, good = diag.project(np.array([[1, 2, 10], [0, 0, -1]]), camera)
        np.testing.assert_allclose(uv[0], [30, 50])
        self.assertFalse(good[1])
        self.assertTrue(np.isnan(uv[1]).all())
        with self.assertRaises(ValueError):
            diag.project(np.array([[np.nan, 0, 1]]), camera)

    def test_triangle_depth_and_head_occlusion(self):
        camera = np.eye(3, 4)
        vertices = np.array([[0, 0, 2], [20, 0, 2], [0, 20, 2]])
        depth = diag.surface_depth(vertices, np.array([[0, 1, 2]]), camera, (16, 16))
        self.assertAlmostEqual(depth[2, 2], 2)
        self.assertTrue(np.isinf(depth[15, 15]))
        # Same image coordinates, behind the triangle surface.
        behind = np.array([[[8, 8, 4], [12, 8, 4]]])
        mask, notes = diag.hair_support(behind, camera, (16, 16), depth)
        self.assertFalse(mask.any())
        self.assertGreater(notes['head_occluded_sample_pixels'], 0)
        front = behind / 4
        mask, _ = diag.hair_support(front, camera, (16, 16), depth)
        self.assertTrue(mask.any())

    def test_metric_detects_displacement_and_radius_sensitivity(self):
        truth = np.zeros((32, 32), bool)
        truth[10:20, 10:20] = True
        support = np.zeros_like(truth)
        support[12:18, 14] = True
        _, a = diag.support_metrics(support, truth, 1)
        _, b = diag.support_metrics(support, truth, 3)
        _, shifted = diag.support_metrics(np.roll(support, 15, axis=1), truth, 1)
        self.assertGreater(b['mask_covered_fraction_proxy'], a['mask_covered_fraction_proxy'])
        self.assertGreater(shifted['mean_centerline_distance_to_mask_px'], a['mean_centerline_distance_to_mask_px'])
        _, empty = diag.support_metrics(np.zeros_like(truth), truth, 1)
        self.assertIsNone(empty['mean_mask_distance_to_centerline_px'])

    def test_obj_preserves_vertices_and_negative_face_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.obj'
            path.write_text('v 0 0 1\nv 1 0 1\nv 0 1 1\nf -3 -2 -1\n')
            vertices, faces = diag.read_obj(path)
            np.testing.assert_array_equal(faces, [[0, 1, 2]])
            self.assertEqual(vertices.shape, (3, 3))

    def test_override_rejects_unsupported_or_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'options.json'
            for bad in ({'loss_config.use_conf': True}, {'loss_config.gaus_depth_loss': float('nan')},
                        {'loss_config.gaus_depth_loss': True}, {'dataset_real.convert_gabor_to_strand_map': 1}):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_overrides(path)
            path.write_text(json.dumps({'loss_config.gaus_depth_loss': 0}))
            self.assertEqual(load_overrides(path), {'loss_config.gaus_depth_loss': 0})

    def test_npz_must_match_recorded_ply_not_just_shape(self):
        import trimesh
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'pointclouds_train').mkdir()
            (run / 'export').mkdir()
            strands = np.array([[[0, 0, 1], [1, 0, 1]]], dtype=np.float32)
            ply = run / 'pointclouds_train/pred_000050.ply'
            trimesh.PointCloud(strands.reshape(-1, 3)).export(ply)
            np.savez(run / 'export/hair.strands.npz', strands=strands)
            report = {'epoch_limit': 50, 'export': {'source_sha256': sha256(ply), 'strand_count': 1, 'points_per_strand': 2}}
            suite.validate_npz(run, report)
            np.savez(run / 'export/hair.strands.npz', strands=strands+1)
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                suite.validate_npz(run, report)

    def test_review_writes_html_diagnostics_and_failure_status(self):
        """Real CPU rendering and export validation; GPU worker deliberately fails."""
        import trimesh
        from pyhocon import ConfigFactory
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            (project / 'scripts').mkdir()
            (project / 'scripts/im2_diagnostics.py').write_bytes(Path(diag.__file__).read_bytes())
            fake_script = project / 'scripts/run_im2_review.py'
            repo = project / 'repo'
            dataset = repo / 'data/personal'
            (dataset / 'resized_img_aligned').mkdir(parents=True)
            Image.new('RGB', (32, 32), 'gray').save(dataset / 'resized_img_aligned/front.png')
            for sub in ('strand_map_aligned', 'orientation_maps_aligned'):
                (dataset / sub).mkdir()
                Image.new('RGB' if sub.startswith('strand') else 'L', (32, 32)).save(dataset / sub / 'front.png')
            image = Image.open(dataset / 'resized_img_aligned/front.png').convert('RGB')
            mask = np.zeros((32, 32), bool); mask[5:15, 5:15] = True
            for file, text in {'configs/static.conf': 'dataset { data_path="./data/" }\ndataset_real { convert_gabor_to_strand_map=false }\nloss_config { gaus_depth_loss=0.01 }',
                               'src/datasets/real_dataset.py': 'if self.convert_gabor_to_strand_map:',
                               'data/head_prior.obj': 'v 0 0 2\nv 60 0 2\nv 0 60 2\nf 1 2 3\n'}.items():
                p = repo / file; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
            canonical = project / 'canonical'
            (canonical / 'bfm_meshes_our_space_aligned').mkdir(parents=True)
            (canonical / 'bfm_meshes_our_space_aligned/front.obj').write_text((repo / 'data/head_prior.obj').read_text())
            case = {'dataset_folder': 'personal'}
            old_reports = {}
            for label, epoch in [('baseline_50', 50), ('baseline_300', 300)]:
                run = project / label; case[label] = str(run)
                (run / 'export').mkdir(parents=True)
                (run / 'pointclouds_train').mkdir()
                strands = np.array([[[6, 6, 1], [12, 6, 1]]], dtype=np.float32)
                np.savez(run / 'export/hair.strands.npz', strands=strands)
                ply = run / ('pointclouds_train/pred_%06d.ply' % epoch)
                trimesh.PointCloud(strands.reshape(-1, 3)).export(ply)
                old_reports[label] = {'epoch_limit': epoch, 'export': {'source_sha256': sha256(ply), 'strand_count': 1, 'points_per_strand': 2}}
            case_path = project / 'case.json'; case_path.write_text(json.dumps(case))
            checked = lambda run, *a: old_reports[run.name]
            audit = {'configs/static.conf': sha256(repo / 'configs/static.conf')}
            values = (dataset, canonical, np.eye(3, 4), image, mask, {}, audit)
            import subprocess
            with patch.object(suite, '__file__', str(fake_script)), \
                 patch.object(sys, 'argv', ['run_im2_review.py', '--case', str(case_path), '--repo', str(repo)]), \
                 patch.object(suite, 'validate_case', return_value=values), \
                 patch.object(suite, 'check_run', side_effect=checked), \
                 patch.object(suite.subprocess, 'check_output', return_value=PIN), \
                 patch.object(suite.subprocess, 'run', return_value=subprocess.CompletedProcess([], 9)):
                self.assertEqual(suite.main(), 1)
            out = next((project / 'runs').iterdir())
            report = json.loads((out / 'suite.report.json').read_text())
            self.assertEqual(report['status'], 'PARTIAL_FAILURE')
            self.assertEqual(report['results'][-1]['exit_code'], 9)
            self.assertTrue((out / 'baseline_50/overlay_r2.png').is_file())
            self.assertIn('No automatic winner', (out / 'review.html').read_text())

    def test_changed_original_photo_rejected_before_running_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            prep = project / 'prep'; prep.mkdir()
            (prep / 'report.json').write_text(json.dumps({'input_sha256': 'wrong'}))
            (project / 'front.jpg').write_bytes(b'changed photo')
            case = {'dataset_folder': 'personal', 'prepared_run': 'prep',
                    'canonical_run': 'deep/canonical', 'front_photo': 'front.jpg'}
            with self.assertRaisesRegex(ValueError, 'does not match'):
                suite.validate_case(case, project, project / 'repo')


if __name__ == '__main__':
    unittest.main()
