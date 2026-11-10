import unittest
import numpy as np
import trimesh
from tools.headapi.neck_print_trial import clip_above, close_boundaries


class NeckClipTests(unittest.TestCase):
    def test_half_box_closes_with_expected_volume_and_preserves_source(self):
        original=trimesh.creation.box(extents=[2,2,2]);vertices=original.vertices.copy()
        clipped=clip_above(original,0);closed,_=close_boundaries(clipped)
        np.testing.assert_array_equal(original.vertices,vertices)
        self.assertTrue(closed.is_volume)
        self.assertAlmostEqual(closed.volume,4.,places=6)
        self.assertGreaterEqual(float(closed.bounds[0,2]),0)

    def test_plane_below_mesh_preserves_volume(self):
        original=trimesh.creation.box(extents=[2,2,2]);clipped=clip_above(original,-2)
        self.assertTrue(clipped.is_volume)
        self.assertAlmostEqual(clipped.volume,original.volume,places=6)
