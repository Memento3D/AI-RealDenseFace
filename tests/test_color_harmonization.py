import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('harmonization',Path(__file__).resolve().parents[1]/'scripts/color_harmonization.py')
color=importlib.util.module_from_spec(spec);spec.loader.exec_module(color)


class ColorTests(unittest.TestCase):
    def test_robust_fit_resists_corrupted_correspondences(self):
        rng=np.random.default_rng(19)
        f=np.column_stack((np.ones(180),rng.uniform(-.5,.5,(180,3))))
        expected=np.array([[.12,.03,-.04],[.04,0,.02],[0,.03,0],[.01,0,0]])
        target=f@expected;target[::9]+=1.5
        fitted=color.solve(f,target)
        error=np.median(np.linalg.norm(f@fitted-f@expected,axis=1))
        self.assertLess(error,.04)

    def test_rgb_conversion_remains_finite_at_zero(self):
        converted=color.rgb(np.array([[0.,.01,.5]]))
        self.assertTrue(np.isfinite(converted).all())
        self.assertEqual(converted[0,0],0)
