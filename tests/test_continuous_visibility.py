import unittest
try:
    import numpy as np
    from tools.headapi.continuous_visibility import ContinuousDepth
except ImportError:
    np=None


@unittest.skipIf(np is None,'Optional numpy unavailable')
class ContinuousVisibilityTests(unittest.TestCase):
    def test_sloped_plane_has_correct_subpixel_perspective_depth(self):
        vertices=np.array([[0.,0.,2.],[8.,0.,4.],[0.,8.,4.]])
        model=ContinuousDepth(vertices,[[0,1,2]],np.eye(3),(8,8),cell_size=1)
        points=np.array([[.25,.25],[.75,.1],[1.1,.3]])
        expected=1/((1-points.sum(1)/2)/2+points.sum(1)/2/4)
        self.assertTrue(np.allclose(model.query(points),expected))
        self.assertNotAlmostEqual(model.query(points[:1])[0],model.query([[.5,.5]])[0])

    def test_front_triangle_occludes_back_and_invalid_points_are_rejected(self):
        vertices=np.array([[0,0,2],[8,0,2],[0,8,2],[0,0,4],[16,0,4],[0,16,4]],float)
        model=ContinuousDepth(vertices,[[0,1,2],[3,4,5]],np.eye(3),(8,8))
        depth=model.query([[.2,.2],[5,5],[-1,0],[float('nan'),1]])
        self.assertAlmostEqual(depth[0],2);self.assertTrue(np.isinf(depth[1:]).all())

    def test_bin_edges_and_degenerate_triangles(self):
        vertices=np.array([[0,0,2],[64,0,2],[0,64,2],[0,0,3],[3,3,3],[6,6,3]],float)
        model=ContinuousDepth(vertices,[[0,1,2],[3,4,5]],np.eye(3),(64,64),cell_size=16)
        self.assertTrue(np.allclose(model.query([[16,0],[15.999,.001],[0,16]]),2))

    def test_positive_depth_required(self):
        with self.assertRaises(ValueError):ContinuousDepth([[0,0,-1],[1,0,1],[0,1,1]],[[0,1,2]],np.eye(3),(8,8))

    def test_nearby_occluder_is_still_nearest_surface(self):
        vertices=np.array([[0,0,2],[8,0,2],[0,8,2],[0,0,2.0005],[8.002,0,2.0005],[0,8.002,2.0005]],float)
        nearest=ContinuousDepth(vertices,[[0,1,2],[3,4,5]],np.eye(3),(8,8)).query([[.3,.3]])[0]
        self.assertAlmostEqual(nearest,2.,places=12)
        self.assertGreater(abs(nearest-2.0005),1e-6)
