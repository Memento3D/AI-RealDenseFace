"""Circular resampling, source association and the one-worker budget guard."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import im2_field_fix as fields
import run_im2_fix as runner
from im2_support import PIN, sha256


class FieldFixTests(unittest.TestCase):
    def test_circular_interpolation_does_not_average_one_and_179_to_90(self):
        theta = np.deg2rad(np.array([[1., 179.]]))
        affine = np.array([[1., 0, .5], [0, 1, 0]])
        angle, strength, good = fields.warp_field(theta, np.ones_like(theta), affine, (2, 1), axial=True)
        delta = abs((angle[0, 1]+np.pi/2) % np.pi-np.pi/2)
        self.assertLess(np.rad2deg(delta), .1)
        self.assertGreater(strength[0, 1], .99)
        scalar = cv2.warpAffine(theta.astype(np.float32), affine, (2, 1))
        self.assertAlmostEqual(np.rad2deg(scalar[0, 1]), 90, places=4)

    def test_similarity_rotates_directed_vectors_and_scale_preserves_direction(self):
        theta = np.zeros((3, 3))
        affine = np.array([[0., -1, 2], [1, 0, 0]])
        angle, _, good = fields.warp_field(theta, np.ones_like(theta), affine, (3, 3), axial=False)
        np.testing.assert_allclose(angle[good], np.pi/2, atol=1e-6)
        for bad in ([[1, .3, 0], [0, 1, 0]], [[-1, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0]]):
            with self.assertRaises(ValueError):
                fields.check_affine(bad)

    def test_correct_fields_preserves_R_and_external_pixels(self):
        mask = np.zeros((8, 8), bool); mask[2:6, 2:6] = True
        raw = np.zeros((8, 8, 3), np.uint8); raw[..., 1] = 255; raw[..., 2] = 128
        old = raw.copy(); old[..., 0] = np.arange(8)[None, :]
        g, strand, note = fields.correct_fields(np.zeros((8, 8), np.uint8), raw, mask, old, mask,
                                                np.array([[1., 0, 0], [0, 1, 0]]))
        np.testing.assert_array_equal(strand[..., 0], old[..., 0])
        np.testing.assert_array_equal(strand[~mask], old[~mask])
        self.assertEqual(note['gabor_support_missing_inside_mask'], 0)
        self.assertFalse(note['camera_modified'])
        with self.assertRaises(ValueError):
            fields.correct_fields(np.zeros((8, 8), np.uint8), raw, np.zeros_like(mask), old, mask,
                                  np.array([[1., 0, 0], [0, 1, 0]]))

    def make_fixture(self, project):
        repo = project / 'repo'
        dataset = repo / 'data/personal'
        prep = project / 'prep'; prep.mkdir()
        size = 32
        mask = np.zeros((size, size), bool); mask[4:16, 4:20] = True
        rgb = np.full((size, size, 3), 100, np.uint8)
        strand = np.full_like(rgb, 128); strand[..., 0] = np.uint8(mask)*255
        strand[..., 1] = 255
        gabor = np.zeros((size, size), np.uint8)
        for folder, data in [('resized_img', rgb), ('seg', np.uint8(mask)*255), ('strand_map', strand), ('orientation_maps', gabor)]:
            raw_folder = prep / folder; raw_folder.mkdir()
            Image.fromarray(data).save(raw_folder / 'front.png')
            aligned = dataset / (folder+'_aligned'); aligned.mkdir(parents=True)
            Image.fromarray(data).save(aligned / 'front.png')
        for folder in ('body_img_aligned', 'confidence_maps_aligned', 'depth_apple_pro_aligned', 'proj_matx_inv_aligned'):
            (dataset / folder).mkdir()
        Image.fromarray(np.uint8(mask)*255).save(dataset / 'body_img_aligned/front.png')
        np.save(dataset / 'confidence_maps_aligned/front.npy', np.ones(mask.shape))
        np.savez(dataset / 'depth_apple_pro_aligned/front.npz', depth=np.ones(mask.shape))
        camera = np.eye(3, 4)
        np.savetxt(dataset / 'proj_matx_inv_aligned/front.txt', camera)
        (repo / 'data/head_prior.obj').write_text('v 0 0 2\nv 60 0 2\nv 0 60 2\nf 1 2 3\n')
        case = {'dataset_folder': 'personal', 'prepared_run': 'prep', 'affine_2d': [[1,0,0],[0,1,0]]}
        reports = {}
        for label, epoch in [('baseline_50', 50), ('baseline_300', 300)]:
            run = project / label; (run / 'export').mkdir(parents=True)
            strands = np.array([[[6, 6, 1], [12, 6, 1], [15, 6, 1]]], np.float32)
            np.savez(run / 'export/hair.strands.npz', strands=strands)
            case[label] = str(run); reports[label] = {'epoch_limit': epoch}
        return repo, dataset, prep, rgb, mask, camera, case, reports

    def test_prepare_copy_rejects_wrong_affine_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            repo, dataset, prep, rgb, mask, camera, case, reports = self.make_fixture(project)
            out = project / 'out'; out.mkdir()
            before = {str(p.relative_to(dataset)): sha256(p) for p in dataset.rglob('*') if p.is_file()}
            target = repo / 'data/fixed'
            with self.assertRaisesRegex(ValueError, 'does not reproduce aligned RGB'):
                runner.prepare_copy(dataset, prep, np.array([[1.,0,4],[0,1,0]]), target, out)
            self.assertFalse(target.exists())
            note = runner.prepare_copy(dataset, prep, np.array([[1.,0,0],[0,1,0]]), target, out)
            for relative, digest in before.items():
                self.assertEqual(sha256(dataset / relative), digest)
                if not relative.startswith(('orientation_maps', 'strand_map')):
                    self.assertEqual(sha256(target / relative), digest)
            self.assertTrue(note['aligned_rgb_reproduced_exactly'])
            self.assertTrue((out / 'fields_before_after.png').is_file())

    def test_prepare_copy_rejects_unrelated_raw_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            repo, dataset, prep, *rest = self.make_fixture(project)
            Image.new('L', (32,32), 80).save(prep / 'orientation_maps/front.png')
            out = project / 'out'; out.mkdir()
            with self.assertRaisesRegex(ValueError, 'Raw Gabor'):
                runner.prepare_copy(dataset, prep, np.array([[1.,0,0],[0,1,0]]), repo / 'data/fixed', out)

    def test_cpu_mode_starts_zero_workers_and_gpu_failure_never_retries(self):
        for cpu_only in (True, False):
            with self.subTest(cpu_only=cpu_only), tempfile.TemporaryDirectory() as folder:
                project = Path(folder)
                repo, dataset, prep, rgb, mask, camera, case, reports = self.make_fixture(project)
                scripts = project / 'scripts'; scripts.mkdir()
                for name in ('run_im2_fix.py', 'im2_field_fix.py', 'im2_diagnostics.py'):
                    source = Path(runner.__file__).with_name(name)
                    (scripts / name).write_bytes(source.read_bytes())
                case_path = project / 'case.json'; case_path.write_text(json.dumps(case))
                values = (dataset, project, camera, Image.fromarray(rgb), mask, {}, {})
                argv = ['run_im2_fix.py', '--case', str(case_path), '--repo', str(repo)]
                if cpu_only: argv.append('--cpu-only')
                with patch.object(runner, '__file__', str(scripts / 'run_im2_fix.py')), \
                     patch.object(sys, 'argv', argv), \
                     patch.object(runner.subprocess, 'check_output', return_value=PIN), \
                     patch.object(runner, 'validate_case', return_value=values), \
                     patch.object(runner, 'check_run', side_effect=lambda run,*a: reports[run.name]), \
                     patch.object(runner, 'check_corrected_sample'), \
                     patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as worker:
                    self.assertEqual(runner.main(), 0 if cpu_only else 1)
                    self.assertEqual(worker.call_count, 0 if cpu_only else 1)
                out = next((project / 'runs').iterdir())
                saved = json.loads((out / 'experiment.report.json').read_text())
                self.assertEqual(saved['new_gpu_runs_started'], 0 if cpu_only else 1)
                self.assertEqual(saved['status'], 'CPU_REVIEW_READY' if cpu_only else 'WORKER_FAILED')
                self.assertTrue((out / 'baseline_50/roots.png').is_file())
                self.assertTrue((out / 'review.html').is_file())
                self.assertEqual(saved['root_position_comparisons']['baseline_300']['max_indexed_displacement_native_units'], 0)


if __name__ == '__main__':
    unittest.main()
