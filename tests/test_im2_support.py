"""Guard against ambiguous upstream patches and strand corruption."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import im2_support as support


class Im2SupportTests(unittest.TestCase):
    def test_monitor_preserves_nonzero_child_exit_and_logs(self):
        from run_im2_sample import monitored
        import os
        if sys.platform != 'linux':
            self.skipTest('Linux process groups')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch('run_im2_sample.gpu_sample', return_value=[None] * 4):
                result = monitored([sys.executable, '-u', '-c',
                    'import time; print("worker failure"); time.sleep(.3); raise SystemExit(3)'],
                    output, os.environ.copy(), output, .2)
            self.assertEqual(result['exit_code'], 3)
            self.assertFalse(result['interrupted'])
            self.assertIsNone(result['peak_device_vram_mib'])
            self.assertIn('worker failure', (output / 'inference.log').read_text())
            self.assertGreaterEqual(result['sample_count'], 1)

    def test_patch_both_blocks_and_idempotence(self):
        source = ('def update():\n    if flag:\n        if other:\n            pass\n'
                  + support.OLD_APP + '\n'
                  + '    if penalty:\n        if enabled:\n' + support.OLD_SDF + '\n')
        changed = support.fixed_text(source)
        self.assertEqual(support.fixed_text(changed), changed)
        self.assertIn('.clone()', changed)
        self.assertNotIn('pred_points_vis[interested_bs', changed)

    def test_unknown_source_does_not_mutate_file(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            path = repo / 'trainers/trainer_single.py'
            path.parent.mkdir()
            content = 'unexpected upstream changes\n'
            path.write_text(content)
            with self.assertRaises(ValueError):
                support.prepare_fixes(repo, apply=True)
            self.assertEqual(path.read_text(), content)
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_strand_layout_rejects_nan_and_incomplete_strands(self):
        import numpy as np
        with self.assertRaises(ValueError):
            support.strand_array(np.zeros((7, 3)), 4)
        points = np.zeros((8, 3))
        points[0, 0] = np.nan
        with self.assertRaises(ValueError):
            support.strand_array(points, 4)

    def test_export_preserves_order_and_separates_strands(self):
        import numpy as np
        import trimesh
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            vertices = np.array([[0,0,0], [1,0,0], [2,0,0], [0,1,0], [1,1,0], [2,1,0]], dtype=np.float32)
            ply = root / 'hair.ply'
            trimesh.PointCloud(vertices).export(ply)
            report = support.export_strands(ply, root / 'export', 3)
            self.assertEqual(report['strand_count'], 2)
            with np.load(root / 'export/hair.strands.npz') as data:
                np.testing.assert_array_equal(data['strands'], vertices.reshape(2, 3, 3))
            lines = (root / 'export/hair.strands.obj').read_text().splitlines()
            self.assertEqual([line for line in lines if line.startswith('l ')], ['l 1 2 3', 'l 4 5 6'])
            with self.assertRaises(FileExistsError):
                support.export_strands(ply, root / 'export', 3)


if __name__ == '__main__':
    unittest.main()
