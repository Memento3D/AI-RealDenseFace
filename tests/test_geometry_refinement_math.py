import unittest
import tempfile
from pathlib import Path
try:
    import numpy as np
    from tools.headapi.geometry_refinement_math import camera_update,mesh_guard,project
except ImportError:
    np=None


@unittest.skipIf(np is None,'Optional image dependencies unavailable')
class GeometryRefinementTests(unittest.TestCase):
    def test_zero_camera_update_preserves_projection_and_mesh(self):
        v=np.array([[0,0,0],[.1,0,.02],[0,.1,.01]],float);p=np.array([[100,0,32,32],[0,100,32,32],[0,0,1,1]],float)
        q=camera_update(p,v,np.zeros(6),.1)
        self.assertTrue(np.allclose(project(v,q),project(v,p)))

    def test_translation_is_bounded_camera_only(self):
        v=np.array([[0,0,0],[.1,0,0],[0,.1,0]],float);p=np.array([[100,0,32,32],[0,100,32,32],[0,0,1,1]],float)
        q=camera_update(p,v,[0,0,0,1,0,0],.1)
        self.assertTrue(np.allclose(project(v,q)[:,0]-project(v,p)[:,0],.05))

    def test_geometry_guard_rejects_normal_flip_and_large_displacement(self):
        v=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float);f=np.array([[0,1,2]])
        self.assertTrue(mesh_guard(v,v.copy(),f,1)['passed'])
        bad=v.copy();bad[:,1:]=bad[:,1:][:,::-1]
        self.assertFalse(mesh_guard(v,bad,f,1)['passed'])
        self.assertFalse(mesh_guard(v,v+.01,f,1)['passed'])

    def test_gray_overlay_is_independent_of_triangle_draw_order(self):
        from PIL import Image
        from tools.headapi.geometry_camera_review import gray_overlay
        v=np.array([[0,0,1],[1,0,1],[0,1,1],[0,0,2],[2,0,2],[0,2,2]],float)
        f=np.array([[0,1,2],[3,4,5]]);p=np.array([[8,0,8,0],[0,8,8,0],[0,0,1,0]],float)
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);Image.new('RGB',(16,16),(180,180,180)).save(d/'photo.png')
            gray_overlay(v,f,p,d/'photo.png',d/'a.png');gray_overlay(v,f[::-1],p,d/'photo.png',d/'b.png')
            self.assertTrue(np.array_equal(np.asarray(Image.open(d/'a.png')),np.asarray(Image.open(d/'b.png'))))
