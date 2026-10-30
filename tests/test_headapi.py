"""API/queue/worker-contract tests use synthetic images and explicitly named doubles.
No neural inference or production print validation is performed by these tests.
"""
import io,json,sys,tempfile,unittest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
from tools.headapi.api import create_app,normalize
from tools.headapi.service import JobService,load_config,write

class HeldExecutor:
 def __init__(self):self.pending=[]
 def submit(self,fn,*args):self.pending.append((fn,args))
 def shutdown(self,**kwargs):pass

def image(color=(150,100,80)):
 stream=io.BytesIO();Image.new('RGB',(80,100),color).save(stream,format='PNG');return stream.getvalue()

class APITests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.executor=HeldExecutor()
  paths={k:str(self.root/k) for k in ('cpu_python','probe_python','rdf_python','landmarker','rdf_repo','cache_root','jobs','parser')}
  for k in ('cpu_python','probe_python','rdf_python','landmarker','parser'):Path(paths[k]).write_bytes(b'fixture, not a model')
  for rel in ('weights/flame/flame2023.pkl','weights/flame/flame_assets.npz','weights/realdenseface/vitb.pth','weights/facebox/face_box.pth'):
   p=Path(paths['rdf_repo'])/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
  self.config={'schema_version':1,'root':str(self.root),'paths':paths,'max_pending_jobs':3}
  self.service=JobService(self.config,self.executor);self.client=TestClient(create_app(self.service));self.client.__enter__()
 def tearDown(self):self.client.__exit__(None,None,None);self.temp.cleanup()
 def post(self,slots=('front',),mode='components'):
  files={slot:(slot+'.png',image((100+i*20,110,120)),'image/png') for i,slot in enumerate(slots)}
  return self.client.post('/v1/head/jobs',files=files,data={'mode':mode})
 def test_mvp_is_default_upload_mode(self):
  r=self.client.post('/v1/head/jobs',files={'front':('front.png',image(),'image/png')})
  self.assertEqual(r.status_code,202,r.text);self.assertEqual(self.service.get(r.json()['job_id'])['mode'],'mvp')
 def test_assembly_creates_child_and_does_not_modify_parent(self):
  id=self.post().json()['job_id'];folder=self.service.jobs/id
  for name in ['rdf','rdf_crops','raw','observation']:(folder/name).mkdir()
  (folder/'rdf/projection_bundle.npz').write_bytes(b'test fixture')
  (folder/'raw/raw_native.glb').write_bytes(b'test fixture')
  write(folder/'rdf/report.json',{'status':'FIT_GATES_PASSED'})
  write(folder/'raw/report.json',{'status':'RAW_PROPOSAL_EXPORTED'})
  write(folder/'rdf_crops/report.json',{});write(folder/'observation/report.json',{})
  self.service.update(id,status='COMPONENTS_EXPORTED');before=(folder/'job.json').read_bytes()
  r=self.client.post('/v1/head/jobs/'+id+'/assemble');self.assertEqual(r.status_code,202,r.text)
  child=self.service.get(r.json()['job_id']);self.assertEqual(child['parent_job_id'],id)
  self.assertEqual(child['job_type'],'assembly_only');self.assertEqual(before,(folder/'job.json').read_bytes())
  self.assertFalse(child['hybrid_assembly_performed'])
 def test_running_parent_cannot_be_assembled(self):
  id=self.post().json()['job_id'];self.assertEqual(self.client.post('/v1/head/jobs/'+id+'/assemble').status_code,409)
 def test_missing_assembly_source(self):
  self.assertEqual(self.client.post('/v1/head/jobs/'+'0'*32+'/assemble').status_code,404)
 def test_front_required(self):self.assertEqual(self.post(('left',)).status_code,422);self.assertFalse(self.executor.pending)
 def test_all_four_supported_view_combinations(self):
  self.config['max_pending_jobs']=10;self.service.config['max_pending_jobs']=10
  for slots in [('front',),('front','left'),('front','right'),('front','left','right')]:
   with self.subTest(slots=slots):
    r=self.post(slots);self.assertEqual(r.status_code,202,r.text)
    row=self.service.get(r.json()['job_id']);self.assertEqual([v['slot'] for v in row['observations']],list(slots))
    self.assertEqual(row['rdf_views_actually_used'],[]);self.assertFalse(row['hybrid_assembly_performed'])
 def test_no_front_duplication_for_missing_sides(self):
  r=self.post();folder=self.service.jobs/r.json()['job_id']/'inputs';self.assertEqual([p.name for p in folder.iterdir()],['front.png'])
 def test_identical_view_pixels_rejected(self):
  self.assertEqual(self.client.post('/v1/head/jobs',files={s:(s+'.png',image(),'image/png') for s in ['front','left']}).status_code,422)
 def test_raw_does_not_silently_accept_unused_sides(self):self.assertEqual(self.post(('front','left'),'raw').status_code,422)
 def test_invalid_image_rejected(self):self.assertEqual(self.client.post('/v1/head/jobs',files={'front':('bad.jpg',b'not-image','image/jpeg')}).status_code,422)
 def test_resolution_rejected(self):self.assertEqual(self.client.post('/v1/head/jobs',files={'front':('a.png',image())},data={'resolution':'512'}).status_code,422)
 def test_queue_capacity(self):
  for _ in range(3):self.assertEqual(self.post().status_code,202)
  self.assertEqual(self.post().status_code,429)
 def test_missing_backend_rejected_without_fake_job(self):
  Path(self.config['paths']['rdf_python']).unlink();self.assertEqual(self.post().status_code,503);self.assertFalse(self.executor.pending)
 def test_original_names_cannot_escape_upload_folder(self):
  r=self.client.post('/v1/head/jobs',files={'front':('../../outside.png',image(),'image/png')});self.assertEqual(r.status_code,202)
  self.assertFalse((self.root/'outside.png').exists())
 def test_status_and_protected_artifacts(self):
  r=self.post();id=r.json()['job_id'];self.assertEqual(self.client.get('/v1/head/jobs/'+id).status_code,200)
  for path in ['inputs/front.png','config.snapshot.json','../job.json']:
   with self.assertRaises(KeyError):self.service.artifact(id,path)
  folder=self.service.jobs/id;(folder/'test.glb').write_bytes(b'synthetic mesh fixture')
  self.service.update(id,artifacts=[{'path':'test.glb'}])
  self.assertEqual(self.client.get('/v1/head/jobs/'+id+'/files/test.glb').content,b'synthetic mesh fixture')
 def test_unknown_job(self):self.assertEqual(self.client.get('/v1/head/jobs/'+'0'*32).status_code,404)
 def test_changed_queued_input_stops_before_worker(self):
  id=self.post(('front',),'raw').json()['job_id'];(self.service.jobs/id/'inputs/front.png').write_bytes(b'changed')
  self.service.stage=lambda *a,**k:self.fail('Must reject before running any stage')
  self.service.run(id);row=self.service.get(id);self.assertEqual(row['status'],'FAILED')
  self.assertIn('Input changed',row['errors'][0]['error'])
 def test_worker_failed_stage_does_not_mark_complete(self):
  id=self.post(('front',),'raw').json()['job_id']
  self.service.stage=lambda *a,**k:False
  self.service.run(id);row=self.service.get(id);self.assertEqual(row['status'],'FAILED');self.assertFalse(row['ready_for_machine'])
 def test_orchestration_synthetic_contract_uses_separate_envs_and_reports_components(self):
  id=self.post(('front','right')).json()['job_id'];folder=self.service.jobs/id;calls=[]
  def stage(job,name,args):
   calls.append((name,args))
   if name=='02_rdf_fit':
    (folder/'rdf').mkdir();write(folder/'rdf/report.json',{'status':'FIT_GATES_PASSED','views_used':['front','right']})
   if name=='04_triposr':
    (folder/'raw').mkdir();write(folder/'raw/report.json',{'status':'RAW_PROPOSAL_EXPORTED'})
   return True
  self.service.stage=stage;self.service.run(id);row=self.service.get(id)
  self.assertEqual(row['status'],'COMPONENTS_EXPORTED');self.assertEqual(row['rdf_views_actually_used'],['front','right'])
  self.assertFalse(row['hybrid_assembly_performed']);self.assertFalse(row['ready_for_machine'])
  self.assertEqual(next(a[0] for n,a in calls if n=='02_rdf_fit'),self.config['paths']['rdf_python'])
  self.assertEqual(next(a[0] for n,a in calls if n=='04_triposr'),self.config['paths']['probe_python'])
 def test_restart_marks_unfinished_attempt_without_rerun(self):
  id=self.post().json()['job_id'];self.service.lock.release();self.service.start()
  self.assertEqual(self.service.get(id)['status'],'INTERRUPTED');self.assertEqual(len(self.executor.pending),1)
 def test_process_lock_prevents_two_servers_for_same_directory(self):
  other=JobService(self.config,HeldExecutor())
  with self.assertRaises(RuntimeError):other.start()
 def test_logs_are_bounded_and_whitelisted(self):
  id=self.post().json()['job_id'];self.service.update(id,current_stage='01_test',stages=[{'name':'01_test'}])
  (self.service.jobs/id/'01_test.log').write_text('x'*100000)
  self.assertEqual(len(self.service.logs(id)),65536)
  with self.assertRaises(KeyError):self.service.logs(id,'../config.snapshot')
 def test_config_paths_resolve_against_project_not_cwd(self):
  p=self.root/'config.json';write(p,{'schema_version':1,'paths':{'cpu_python':'env/python'}})
  self.assertEqual(load_config(p,self.root)['paths']['cpu_python'],str(self.root/'env/python'))

class FaceHelpersTests(unittest.TestCase):
 def test_available_slots_never_fabricates_side_photos(self):
  from tools.headapi.prepare_faces import available_slots
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'front.png').write_bytes(b'x');self.assertEqual(available_slots(root),['front'])
   (root/'right.jpg').write_bytes(b'y');self.assertEqual(available_slots(root),['front','right'])
 def test_face_crop_can_pad_without_deforming_projection(self):
  import numpy as np
  from tools.headapi.prepare_faces import face_crop
  from head24_policy import apply_pixels
  crop,matrix=face_crop([1,2,81,102],(100,110));xy=apply_pixels(np.array([[1,2],[81,102]]),matrix)
  self.assertEqual(matrix.shape,(3,3));self.assertTrue((xy>=0).all());self.assertTrue((xy<=1024).all());self.assertLess(crop[0],0)
 def test_export_preserves_original_geometry(self):
  import numpy as np,trimesh
  from tools.headapi.face_export import export_face
  from tools.triposr_probe.audit import load_mesh
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);m=trimesh.creation.icosphere(subdivisions=1);v=m.vertices.copy();v[:,2]+=4
   P=np.array([[60,0,40,0],[0,60,50,0],[0,0,1,0]],float)
   np.savez(root/'bundle.npz',vertices=v,faces=m.faces,view_vertices=np.array([v]),cameras=np.array([P]))
   (root/'front.png').write_bytes(image())
   result=export_face(root/'bundle.npz',root,['front'],root)
   np.testing.assert_allclose(load_mesh(root/'rdf_face_gray.glb').vertices,v,atol=1e-6)
   self.assertTrue(result['geometry_preserved_from_bundle']);self.assertFalse(result['ready_for_machine'])



class FittingContractTests(unittest.TestCase):
 def test_no_joint_registration_for_front_only(self):
  import numpy as np
  from unittest.mock import Mock
  from tools.headapi.fitting_contract import fit_shared_identity
  optimizer=Mock();a=np.zeros((1,4,2));camera=np.zeros((1,1,4,4));states=np.zeros((1,118));identity=object()
  b,x=fit_shared_identity(optimizer,camera,camera,a,a,a,a,identity,states,1)
  self.assertIs(b,identity);self.assertIs(x,states);optimizer._register.assert_not_called()
 def test_two_and_three_views_keep_their_real_batch_sizes(self):
  import numpy as np
  from unittest.mock import Mock
  from tools.headapi.fitting_contract import fit_shared_identity
  for count in (2,3):
   optimizer=Mock();a=np.zeros((count,4,2));camera=np.zeros((count,1,4,4));states=np.zeros((count,118));identity=object()
   optimizer._register.return_value=('beta','states')
   self.assertEqual(fit_shared_identity(optimizer,camera,camera,a,a,a,a,identity,states,count),('beta','states'))
   self.assertIs(optimizer._register.call_args.kwargs['init_identity'],identity)
   self.assertEqual(optimizer._register.call_args.args[2].shape[0],count)
 def test_wrong_view_count_rejected(self):
  import numpy as np
  from tools.headapi.fitting_contract import fit_shared_identity
  a=np.zeros((1,4,2));camera=np.zeros((3,1,4,4))
  with self.assertRaises(ValueError):fit_shared_identity(None,camera,camera,a,a,a,a,None,a,1)

if __name__=='__main__':unittest.main()
