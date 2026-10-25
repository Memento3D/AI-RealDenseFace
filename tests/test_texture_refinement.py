import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('boundary',SCRIPTS/'refine_texture_boundary.py')
boundary=importlib.util.module_from_spec(spec);spec.loader.exec_module(boundary)


class BoundaryTests(unittest.TestCase):
    def test_masked_triangle_stays_unprojected(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[1,0,0],[1,1,0],[0,1,0]],dtype=float)
        original=vertices.copy();weights=np.zeros((2,3,3));weights[0,:,0]=1
        coords=np.ones((3,2,3,3));skin=np.array([.6,.4,.3])
        def sample(image,uv):return np.tile(skin,(len(uv),1))
        alpha,fill,report=boundary.prepare(vertices,weights,coords,[None]*3,np.ones((3,3)),skin,sample,lambda x:x)
        self.assertTrue((alpha[1]==0).all())
        self.assertTrue(np.isfinite(fill).all())
        self.assertTrue(((fill>=0)&(fill<=1)).all())
        np.testing.assert_array_equal(vertices,original)
        self.assertFalse(report['new_photo_projection'])
        self.assertEqual(report['boundary_vertices'],2)
