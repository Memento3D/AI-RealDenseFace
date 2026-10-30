import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from PIL import Image
from tools.headlab26.io import dataset_ids,inspect_dataset,key,seal_stage,reuse_stage,write,read
from tools.headlab26.policy import observation_confidence,adjusted_uncertainty,robust_confidence,eye_aperture
from tools.headlab26.geometry import arap

class DatasetTests(unittest.TestCase):
 def images(self,folder):
  folder.mkdir(parents=True)
  for i,name in enumerate(['front','left','right']):Image.new('RGB',(90+i*3,100),(80+i*30,60,50)).save(folder/(name+'.png'))
 def test_range_and_sparse_ids(self):self.assertEqual(dataset_ids('01-03,06,12,02'),['01','02','03','06','12'])
 def test_unsafe_or_descending_ids_rejected(self):
  for s in ['../01','04-01','01-1000']:
   with self.assertRaises(ValueError):dataset_ids(s)
 def test_three_inputs_have_content_fingerprint(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d)/'01';self.images(folder);a=inspect_dataset('01',folder);self.assertEqual(a['status'],'INPUTS_DISCOVERED');self.assertEqual(len(a['views']),3)
   Image.new('RGB',(90,100),(200,100,80)).save(folder/'front.png');b=inspect_dataset('01',folder);self.assertNotEqual(a['input_fingerprint'],b['input_fingerprint'])
 def test_same_photos_in_another_dataset_cannot_share_key(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d)/'01';self.images(folder);self.assertNotEqual(inspect_dataset('01',folder)['input_fingerprint'],inspect_dataset('02',folder)['input_fingerprint'])
 def test_duplicate_images_are_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d)/'01';self.images(folder);(folder/'right.png').write_bytes((folder/'left.png').read_bytes());self.assertIn('DUPLICATE_REQUIRED_VIEW_CONTENT',inspect_dataset('01',folder)['errors'])
 def test_duplicate_extension_is_not_arbitrarily_selected(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d)/'01';self.images(folder);Image.new('RGB',(100,100)).save(folder/'front.jpg');self.assertIn('front:EXPECTED_EXACTLY_ONE_IMAGE',inspect_dataset('01',folder)['errors'])
 def test_mapping_cannot_escape_dataset(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d)/'01';self.images(folder);write(folder/'headlab.json',{'views':{'front':'../other.png'}})
   with self.assertRaisesRegex(ValueError,'escapes'):inspect_dataset('01',folder)
 def test_back_is_not_claimed_as_calibrated(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d)/'06';self.images(folder);Image.new('RGB',(100,100),(40,40,40)).save(folder/'back.png');r=inspect_dataset('06',folder);self.assertEqual(r['status'],'INPUTS_DISCOVERED');self.assertIn('UNCALIBRATED',r['warnings'][0])
 def test_missing_directory_remains_a_case_error(self):
  r=inspect_dataset('06',Path('/nonexistent-headlab26-fixture'));self.assertEqual(r['status'],'INPUT_INVALID')

class CacheTests(unittest.TestCase):
 def test_reuse_is_bound_and_checks_every_file(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'source';source.mkdir();write(source/'report.json',{'value':1});seal_stage(source,'data01')
   self.assertFalse(reuse_stage(source,root/'wrong','data02'));self.assertTrue(reuse_stage(source,root/'copy','data01'))
   (source/'report.json').write_text('changed')
   with self.assertRaisesRegex(ValueError,'TAMPERED'):reuse_stage(source,root/'tampered','data01')
 def test_empty_binding_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'src';source.mkdir();write(source/'stage.binding.json',{'key':'k','files':{}})
   with self.assertRaisesRegex(ValueError,'EMPTY'):reuse_stage(source,root/'dst','k')
 def test_output_is_never_overwritten(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);write(p/'report.json',{'ok':True});seal_stage(p,'k')
   with self.assertRaises(ValueError):reuse_stage(p,p,'k')

class ObservationPolicyTests(unittest.TestCase):
 settings={'glasses_weight':.06,'nonface_weight':.15,'outside_weight':.02}
 def test_occluded_and_outside_points_receive_less_trust(self):
  labels=np.ones((10,10),int);labels[2,2]=6;labels[3,3]=17;c,classes=observation_confidence([[1,1],[2,2],[3,3],[-2,1]],labels,self.settings);np.testing.assert_allclose(c,[1,.06,.15,.02])
 def test_visible_eyes_only_receive_boost(self):
  cv=np.zeros((3,1));dv=cv.copy();a,b=adjusted_uncertainty(cv,dv,[1,.06,1],[0,1],-2);self.assertEqual(a[0,0],-2);self.assertGreater(a[1,0],0);np.testing.assert_array_equal(cv,np.zeros((3,1)))
 def test_zero_nan_confidence_refuses(self):
  for confidence in [[1,0],[1,np.nan]]:
   with self.assertRaises(ValueError):adjusted_uncertainty(np.zeros((2,1)),np.zeros((2,1)),confidence,[0])
 def test_huber_does_not_amplify_inliers(self):
  w,r=robust_confidence([[0,0],[30,0]],[[0,0],[0,0]],3);np.testing.assert_allclose(w,[1,.1])
 def test_lid_ratio_is_invariant_to_image_resize(self):
  points=np.array([[0,0],[10,0],[10,2],[0,2],[5,1]]);self.assertAlmostEqual(eye_aperture(points),eye_aperture(points*4+20))

class ARAPTests(unittest.TestCase):
 def mesh(self):
  import trimesh
  m=trimesh.creation.box();return m.vertices.copy(),m.faces.copy()
 def test_identity_and_source_vertices_unchanged(self):
  p,f=self.mesh();original=p.copy();q,r=arap(p,f,[0,1,2],p[[0,1,2]],8,1);np.testing.assert_allclose(q,p,atol=1e-8);np.testing.assert_array_equal(p,original)
 def test_preserves_rigid_translation(self):
  p,f=self.mesh();delta=np.array([.1,.05,.03]);q,r=arap(p,f,[0,1,2,4],p[[0,1,2,4]]+delta,20,1);np.testing.assert_allclose(q,p+delta,atol=.002);self.assertLessEqual(r['energy_trace'][-1],r['energy_trace'][0]+1e-9)
 def test_rejects_unbounded_move(self):
  p,f=self.mesh()
  with self.assertRaisesRegex(ValueError,'bounded'):arap(p,f,[0,1,2],p[[0,1,2]]+10,3,.1)

class ManufacturingTests(unittest.TestCase):
 def test_actual_layers_flag_floating_component(self):
  from tools.headlab26.manufacturing import check_layers
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a=np.zeros((10,10),np.uint8);a[2:4,2:4]=255;Image.fromarray(a).save(p/'000.png');a[7:9,7:9]=255;Image.fromarray(a).save(p/'001.png');r=check_layers(p);self.assertEqual(r['island_layer_flags'][0]['layer'],1);self.assertFalse(r['physical_print_verified'])
 def test_missing_slicer_never_claims_success(self):
  from tools.headlab26.manufacturing import run
  with tempfile.TemporaryDirectory() as d:
   r=run('unused.stl',Path(d)/'output',{},Path(d));self.assertEqual(r['status'],'SLICER_NOT_CONFIGURED');self.assertFalse(r['ready_for_machine'])
 def test_no_layers_rejected(self):
  from tools.headlab26.manufacturing import check_layers
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):check_layers(d)

class BatchTests(unittest.TestCase):
 images=DatasetTests.images
 def test_six_case_preflight_continues_and_never_launches_inference(self):
  from tools.headlab26 import cli
  from tools.headlab26.config import load
  profile=load(ROOT);profile['resolved']['project']='unused'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);inputs=p/'input'
   for name in ['01','02','03','04','06']:self.images(inputs/name)
   (inputs/'01/right.png').unlink();(inputs/'03/right.png').write_bytes((inputs/'03/left.png').read_bytes())
   out=p/'runs'
   with patch.object(cli,'load',return_value=copy.deepcopy(profile)),patch.object(cli,'verify_code'),patch.object(cli,'execute_python',side_effect=AssertionError('Inference must not run')):
    code=cli.main(['--input-root',str(inputs),'--datasets','01-06','--check-only','--output',str(out)])
   report=read(out/'batch.report.json');self.assertEqual(code,2);self.assertEqual(len(report['datasets']),6);self.assertTrue((out/'index.html').is_file());self.assertEqual(report['datasets'][1]['status'],'PREFLIGHT_ONLY');self.assertEqual(report['datasets'][4]['status'],'INPUT_INVALID')


class VariantSelectionTests(unittest.TestCase):
 def test_occluded_eyes_do_not_win_from_a_misleading_proxy(self):
  from tools.headlab26.cli import choose_variants
  a={'backend':'rdf_baseline','diagnostics':{'eyes':[{'eye_semantic_pixels':0,'glasses_pixels':50}],'eye_contour_error_proxy_median':.3}}
  b=copy.deepcopy(a);b['backend']='rdf_occlusion';b['diagnostics']['eye_contour_error_proxy_median']=.001
  chosen,reason=choose_variants([b,a],True);self.assertEqual(chosen,[a]);self.assertIn('occluded',reason)
 def test_visible_eye_proxy_can_select_a_candidate(self):
  from tools.headlab26.cli import choose_variants
  a={'backend':'rdf_baseline','diagnostics':{'eyes':[{'eye_semantic_pixels':25,'glasses_pixels':0}],'eye_contour_error_proxy_median':.3}}
  b=copy.deepcopy(a);b['backend']='rdf_occlusion';b['diagnostics']['eye_contour_error_proxy_median']=.01
  self.assertEqual(choose_variants([a,b],True)[0],[b])

class AppearanceBindingTests(unittest.TestCase):
 def test_scale_and_semantic_observation_are_bound(self):
  from tools.headlab26.face_states import appearance_key
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'source').write_text('workspace');(root/'semantic').mkdir();write(root/'semantic/report.json',{'labels':'01'})
   profile={name:{} for name in ['appearance23','skin245','glasses245','bright_skin']};profile.update(head_neck_mm=22.,semantic26=str(root/'semantic'))
   old=appearance_key(root/'source',profile);profile['head_neck_mm']=23.;self.assertNotEqual(old,appearance_key(root/'source',profile));profile['head_neck_mm']=22.;write(root/'semantic/report.json',{'labels':'02'});self.assertNotEqual(old,appearance_key(root/'source',profile))


class PixelCameraTests(unittest.TestCase):
 def test_matches_upstream_screen_projection_and_pixel_centres(self):
  from tools.headlab26.pixel_export import pixel_camera
  frame={'flame':{'R_rotation_matrix':np.eye(3)[None],'t':np.array([[.1,.2,0.]])},'camera':{'R_base_0':np.eye(3)[None],'t_base_0':np.array([[0.,0.,-2.]]),'fl':np.array([[1.3]]),'pp':np.array([[.03,-.02]])},'img_size':np.array([512,512])}
  p=np.array([[.1,.2,.3],[-.1,.05,.2]]);K,E=pixel_camera(frame);cv=p@E[:3,:3].T+E[:3,3];actual=cv@K.T;actual=actual[:,:2]/actual[:,2:]
  gl=p+np.array([.1,.2,-2.]);prime=gl/-gl[:,2:];intr=np.eye(3);intr[0,0]=intr[1,1]=1.3*512;intr[:2,2]=256.5+np.array([.03,-.02])*256.5;intr[0,2]=512-intr[0,2]
  native=-(prime@intr.T)[:,:2];native[:,0]=511-native[:,0];expected=(native+.5)*2-.5
  np.testing.assert_allclose(actual,expected,atol=1e-10)


class ControlGraphTests(unittest.TestCase):
 def test_nearby_shell_layers_never_share_a_control_node(self):
  from tools.headlab26.geometry import control_graph
  p=np.array([[0,0,0],[.02,0,0],[0,.02,0],[0,0,.01],[.02,0,.01],[0,.02,.01]])
  graph=control_graph(p,[[0,1,2],[3,4,5],[0,1,4],[0,4,3]],[0,0,0,2,2,2],.5,32)
  self.assertEqual(len(graph['vertices']),2);self.assertNotEqual(graph['inverse'][0],graph['inverse'][3]);np.testing.assert_array_equal(graph['edges'],[[0,1]])
 def test_repeated_control_graph_is_deterministic_and_bounded(self):
  import trimesh
  from tools.headlab26.geometry import control_graph
  mesh=trimesh.creation.icosphere(subdivisions=3);a=control_graph(mesh.vertices,mesh.faces,np.zeros(len(mesh.vertices)),.1,128);b=control_graph(mesh.vertices,mesh.faces,np.zeros(len(mesh.vertices)),.1,128)
  self.assertLessEqual(len(a['vertices']),128);np.testing.assert_array_equal(a['inverse'],b['inverse']);np.testing.assert_allclose(a['vertices'],b['vertices'])
 def test_transfer_preserves_distinct_constant_layer_translations(self):
  from tools.headlab26.geometry import transfer_displacement
  p=np.array([[0,0,0],[.01,0,0],[0,0,.01],[.01,0,.01]])
  graph={'vertices':np.array([[0,0,0],[0,0,.01]]),'layers':np.array([0,2])};delta=np.array([[1,0,0],[0,2,0]])
  q=transfer_displacement(p,[0,0,2,2],graph,graph['vertices']+delta);np.testing.assert_allclose(q,p+delta[[0,0,1,1]])
 def test_all_handle_case_and_unanchored_component(self):
  from tools.headlab26.geometry import arap_edges
  p=np.array([[0.,0,0],[1,0,0],[0,1,0],[10,0,0],[11,0,0],[10,1,0]])
  edges=[[0,1],[1,2],[2,0],[3,4],[4,5],[5,3]]
  with self.assertRaisesRegex(ValueError,'no handle'):arap_edges(p,edges,[0,1,2],p[[0,1,2]])
  q,note=arap_edges(p,edges,np.arange(6),p+.1);np.testing.assert_allclose(q,p+.1);self.assertEqual(note['iterations'],0)
 def test_orientation_backtracking_rejects_fold_without_changing_source(self):
  from tools.headlab26.geometry import orientation_blend
  p=np.array([[0.,0,0],[1,0,0],[0,1,0]]);candidate=p.copy();candidate[2,1]=-1
  original=p.copy();q,note=orientation_blend(p,candidate,[[0,1,2]]);self.assertLess(note['retained_arap_fraction'],1);self.assertGreater(q[2,1],0);np.testing.assert_array_equal(p,original)
 def test_residual_filter_keeps_locked_contact_exact(self):
  from tools.headlab26.geometry import smooth_residual
  p=np.array([[0.,0,0],[1,0,0],[0,1,0]]);q=smooth_residual(p,p+.5,[[0,1,2]],[0],8);np.testing.assert_array_equal(q[0],p[0]);self.assertGreater(np.linalg.norm(q[1]-p[1]),0)

class StageSourceBindingTests(unittest.TestCase):
 def test_hair_edit_reuses_fit_but_policy_edit_invalidates_fit(self):
  from tools.headlab26.cache import stage_code,STAGE_FILES
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for name in set(sum(STAGE_FILES.values(),[])):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
   fit=stage_code(root,'fit');prep=stage_code(root,'prepare');(root/'tools/headlab26/geometry.py').write_text('new hair solver')
   self.assertEqual(stage_code(root,'fit'),fit);self.assertEqual(stage_code(root,'prepare'),prep)
   (root/'tools/headlab26/policy.py').write_text('changed observations');self.assertNotEqual(stage_code(root,'fit'),fit);self.assertEqual(stage_code(root,'prepare'),prep)

if __name__=='__main__':unittest.main()
