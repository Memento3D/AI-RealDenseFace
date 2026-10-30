"""Mathematical/visibility tests for MVP assembly; not likeness or manufacturing tests."""
import unittest,tempfile
from pathlib import Path
import numpy as np
from tools.headapi.fusion_geometry import raster,similarity,sample_surface,transform_points

class FusionGeometryTests(unittest.TestCase):
 def test_positive_similarity_recovers_rotation_scale_translation(self):
  x=np.random.default_rng(1).normal(size=(40,3));a=.4;r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
  y=x@(r*2.1).T+[1,2,3];m,e=similarity(x,y)
  np.testing.assert_allclose(transform_points(x,m),y,atol=1e-10);self.assertGreater(np.linalg.det(m[:3,:3]),0)
 def test_outliers_do_not_create_unbounded_alignment(self):
  x=np.random.default_rng(4).normal(size=(40,3));y=x*.35+[.1,.2,.3];y[-2:]+=2
  m,e=similarity(x,y);self.assertLess(np.median(e),.03)
 def test_degenerate_controls_rejected(self):
  with self.assertRaises(ValueError):similarity(np.zeros((8,3)),np.zeros((8,3)))
 def test_orthographic_front_uses_nearest_triangle_and_actual_points(self):
  v=np.array([[-1,-1,0],[1,-1,0],[0,1,0],[-1,-1,1],[1,-1,1],[0,1,1]],float)
  f=np.array([[0,1,2],[3,4,5]]);colors=np.array([[255,0,0]]*3+[[0,255,0]]*3)
  image,points,ids=raster(v,f,colors,64);self.assertEqual(ids[32,32],1);self.assertAlmostEqual(points[32,32,2],1);self.assertEqual(image[32,32,1],255)
 def test_perspective_interpolation_projects_back_to_same_pixel(self):
  v=np.array([[-1,-1,3],[1,-1,5],[0,1,4]],float);f=np.array([[0,1,2]]);p=np.array([[40,0,32,0],[0,40,32,0],[0,0,1,0]])
  _,points,ids=raster(v,f,np.full((3,3),180),64,camera=p);ys,xs=np.where(ids>=0);k=len(xs)//2;q=p@np.r_[points[ys[k],xs[k]],1]
  np.testing.assert_allclose(q[:2]/q[2],[xs[k]+.5,ys[k]+.5],atol=1e-5)
 def test_surface_sampling_does_not_fabricate_missing_points(self):
  p=np.full((20,20,3),np.nan);points,valid=sample_surface(p,[[5,5]])
  self.assertFalse(valid[0]);self.assertTrue(np.isfinite(points).all())



class FusionPreservationTests(unittest.TestCase):
 def test_lit_preview_does_not_move_geometry_or_recolor_export(self):
  import trimesh
  from tools.headapi.fusion_geometry import preview_assembly
  mesh=trimesh.creation.icosphere(subdivisions=0);mesh.visual.vertex_colors=[180,120,100,255]
  v=mesh.vertices.copy();f=mesh.faces.copy();c=mesh.visual.vertex_colors.copy()
  with tempfile.TemporaryDirectory() as d:preview_assembly(mesh,Path(d)/'preview.png',size=64)
  np.testing.assert_array_equal(mesh.vertices,v);np.testing.assert_array_equal(mesh.faces,f);np.testing.assert_array_equal(mesh.visual.vertex_colors,c)

if __name__=='__main__':unittest.main()
