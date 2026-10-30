"""Rear-observation guards, raster equivalence, fusion and support prechecks."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
 import numpy as np,trimesh
 from PIL import Image,ImageDraw
 from headlab_backview import segment_rear,camera_matrix
 from headlab_raster import surface_depth_fast,render_fast
 from im2_diagnostics import surface_depth
 from headlab_support5 import frustum,scan_layers,candidate_orientations
 from headlab_fusion5 import fused_surface
 from headlab_algorithms import rbf_basis
 from run_headlab_v5 import nominated_candidate
 from figurine_preview import render
 from run_headlab_v4 import save_gray
 AVAILABLE=True
except ImportError:AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'optional HeadLab CPU stack unavailable')
class HeadLabV5Tests(unittest.TestCase):
 def test_dark_rear_segmentation_excludes_skin_and_background(self):
  im=Image.new('RGB',(320,240),(220,220,220));d=ImageDraw.Draw(im)
  d.rectangle((128,145,192,210),fill=(120,80,65));d.ellipse((100,30,220,165),fill=(15,16,18))
  d.rectangle((94,105,104,150),fill=(120,80,65));d.rectangle((216,105,226,150),fill=(120,80,65))
  image,mask,label,report=segment_rear(im)
  self.assertTrue(mask[80,160]);self.assertFalse(mask[190,160]);self.assertFalse(mask[20,20]);self.assertEqual(len(report['ear_region_anchors_image_px']),2)
 def test_uniform_unsupported_input_is_rejected(self):
  with self.assertRaises(ValueError):segment_rear(Image.new('RGB',(320,240),(200,200,200)))
 def test_rear_camera_projects_origin_to_principal_point(self):
  m=camera_matrix([8,4,-2,np.log(12),230,180],(512,300),768);p=m@np.array([0,0,0,1])
  np.testing.assert_allclose(p[:2]/p[2],[230,180]);self.assertGreater(p[2],0)
 def test_fast_depth_matches_reference(self):
  mesh=trimesh.creation.icosphere(subdivisions=2);camera=np.array([[30,0,32,160],[0,-30,32,160],[0,0,1,5.]])
  a=surface_depth(mesh.vertices,mesh.faces,camera,(64,64));b=surface_depth_fast(mesh.vertices,mesh.faces,camera,(64,64))
  np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));np.testing.assert_allclose(a[np.isfinite(a)],b[np.isfinite(b)],atol=1e-10)
 def test_fast_preview_matches_reference(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'mesh.glb';save_gray(path,trimesh.creation.icosphere(subdivisions=2))
   a=np.asarray(render(path,65,100),float);b=np.asarray(render_fast(path,65,100),float);self.assertLess(float(abs(a-b).mean()),.05)
 def test_support_frustum_is_positive_closed(self):
  m=frustum(.15,.3,.8);self.assertTrue(m.is_volume);self.assertTrue(m.is_watertight)
 def test_island_budget_is_not_silently_ignored(self):
  volume=np.zeros((8,8,4),bool);volume[1:3,1:3,0:2]=True;volume[5:7,5:7,2:]=True
  points,unresolved=scan_layers(volume,1,np.zeros(3),.1,np.eye(4),False)
  self.assertEqual(len(points),1);self.assertGreater(len(unresolved),0)
 def test_yaw_changes_gravity_relative_to_head(self):
  rows=candidate_orientations(trimesh.creation.box(extents=[1,2,3]))
  a=next(r for r in rows if r[1:3]==(35,0));b=next(r for r in rows if r[1:3]==(35,180))
  self.assertFalse(np.allclose(a[3][2,:3],b[3][2,:3]))
 def test_fourth_view_adds_posterior_observations(self):
  sphere=trimesh.creation.icosphere(subdivisions=1,radius=2)
  front=np.array([[12,0,-32,160],[0,-12,-32,160],[0,0,-1,5.]]);back=np.array([[-12,0,32,160],[0,-12,32,160],[0,0,1,5.]])
  masks=[];depths=[];labels=[]
  for k,camera in enumerate([front,front,front,back]):
   mask=np.indices((64,64))[0]<(32 if k<3 else 35);masks.append(mask);labels.append(np.where(mask,17,1).astype(np.uint8));depths.append(surface_depth_fast(sphere.vertices,sphere.faces,camera,(64,64)))
  source=sphere.vertices[sphere.faces].reshape(-1,3)
  shared={'source_mm':source,'components':np.zeros(len(sphere.faces),int),'closed':sphere,'cameras':[front,front,front,back],'masks':masks,'depths':depths,'labels':labels,'images':[Image.new('RGB',(64,64))]*4}
  original=sphere.vertices.copy()
  with tempfile.TemporaryDirectory() as directory:
   a=fused_surface(shared,3,Path(directory)/'a',{'back_weight':2,'boundary_regularization':.2});b=fused_surface(shared,4,Path(directory)/'b',{'back_weight':2,'boundary_regularization':.2})
   self.assertLess(b['fusion']['unknown_vertices'],a['fusion']['unknown_vertices']);self.assertTrue(b['fusion']['hairline_recomputed'])
  np.testing.assert_array_equal(sphere.vertices,original)
 def test_compact_basis_is_finite_partition_of_unity(self):
  mesh=trimesh.creation.icosphere(subdivisions=1)
  basis,centers=rbf_basis(mesh.vertices,np.ones(len(mesh.vertices),bool),10,'compact')
  np.testing.assert_allclose(basis.sum(1),1,atol=1e-12)
  self.assertTrue(np.isfinite(basis).all());self.assertTrue((basis==0).any())
 def test_nomination_rejects_large_front_loss(self):
  def row(name,front,back):return {'id':name,'variant':name,'status':'EXPERIMENT_EXPORTED','geometry':{'topology_pass':True},'fit':{'converged':True},'metrics':{'view_ious':{'front':front,'back':back},'common_three_view_iou':front}}
  baseline=row('baseline',.9,.6);bad=row('bad',.75,.99)
  self.assertEqual(nominated_candidate([baseline,bad],.025)['id'],'baseline')
 def test_no_topology_pass_means_no_nomination(self):
  row={'status':'EXPERIMENT_EXPORTED','geometry':{'topology_pass':False}}
  self.assertIsNone(nominated_candidate([row],.025))
if __name__=='__main__':unittest.main()
