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
  paths={k:str(self.root/k) for k in ('cpu_python','probe_python','rdf_python','haircap_python','landmarker','rdf_repo','cache_root','jobs','parser')}
  for k in ('cpu_python','probe_python','rdf_python','haircap_python','landmarker','parser'):Path(paths[k]).write_bytes(b'fixture, not a model')
  for rel in ('weights/flame/flame2023.pkl','weights/flame/flame_assets.npz','weights/realdenseface/vitb.pth','weights/facebox/face_box.pth'):
   p=Path(paths['rdf_repo'])/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
  self.config={'schema_version':1,'root':str(self.root),'paths':paths,'max_pending_jobs':3}
  self.service=JobService(self.config,self.executor);self.client=TestClient(create_app(self.service,api_key='test-backend-key'),headers={'X-API-Key':'test-backend-key'});self.client.__enter__()
 def tearDown(self):self.client.__exit__(None,None,None);self.temp.cleanup()
 def post(self,slots=('front','left','right'),mode='mvp'):
  files={slot:(slot+'.png',image((100+i*20,110,120)),'image/png') for i,slot in enumerate(slots)}
  return self.client.post('/v1/head/jobs',files=files,data={'mode':mode})
 def test_mvp_is_default_upload_mode(self):
  files={slot:(slot+'.png',image((100+i*20,110,120)),'image/png') for i,slot in enumerate(('front','left','right'))}
  r=self.client.post('/v1/head/jobs',files=files)
  self.assertEqual(r.status_code,202,r.text);self.assertEqual(self.service.get(r.json()['job_id'])['mode'],'mvp')
 def test_backend_contract_accepts_three_referenced_images(self):
  from unittest.mock import patch
  from uuid import uuid4
  views=('front','left','right')
  entries=[normalize(image((80+i*40,100,120))) for i in range(len(views))]
  request={'job_id':str(uuid4()),'attempt_id':str(uuid4()),'profile':'default',
           'input_images':[{'image_id':str(uuid4()),'input_url':'https://res.cloudinary.com/demo/image/upload/'+str(i)+'.jpg',
                            'view_hint':view} for i,view in enumerate(views)]}
  with patch('tools.headapi.api._fetch_remote_image',side_effect=entries):
   response=self.client.post('/jobs',json=request,headers={'X-API-Key':'test-backend-key'})
  self.assertEqual(response.status_code,202,response.text)
  self.assertEqual(response.json()['job_id'],request['job_id'])
  self.assertEqual(response.json()['attempt_id'],request['attempt_id'])
  self.assertEqual(response.json()['status'],'queued')
  row=self.service.get(request['job_id'])
  self.assertEqual([item['view_hint'] for item in row['observations']],list(views))
  self.assertEqual(row['observations'][0]['image_id'],request['input_images'][0]['image_id'])
  with patch('tools.headapi.api._fetch_remote_image') as fetch:
   retry=self.client.post('/jobs',json=request)
  self.assertEqual(retry.status_code,202);self.assertEqual(fetch.call_count,0)
 def test_backend_contract_rejects_fewer_than_three_images_before_fetching(self):
  from unittest.mock import patch
  from uuid import uuid4
  for views in (('front',),('front','left')):
   request={'job_id':str(uuid4()),'attempt_id':str(uuid4()),
            'input_images':[{'image_id':str(uuid4()),'input_url':'https://res.cloudinary.com/demo/'+view+'.jpg',
                             'view_hint':view} for view in views]}
   with self.subTest(views=views),patch('tools.headapi.api._fetch_remote_image') as fetch:
    response=self.client.post('/jobs',json=request)
    self.assertEqual(response.status_code,422,response.text)
    fetch.assert_not_called()
    self.assertFalse(self.executor.pending)
 def test_backend_contract_accepts_four_distinct_view_hints(self):
  from unittest.mock import patch
  from uuid import uuid4
  entries=[normalize(image((20+i*40,90,120))) for i in range(4)]
  request={'job_id':str(uuid4()),'attempt_id':str(uuid4()),'profile':'default',
           'input_images':[{'image_id':str(uuid4()),'input_url':'https://res.cloudinary.com/demo/'+str(i),
                            'view_hint':view} for i,view in enumerate(('front','left','right','back'))]}
  with patch('tools.headapi.api._fetch_remote_image',side_effect=entries):
   response=self.client.post('/jobs',json=request)
  self.assertEqual(response.status_code,202,response.text)
  self.assertEqual([row['slot'] for row in self.service.get(request['job_id'])['observations']],
                   ['front','left','right','back'])
 def test_health_status_matches_backend_client(self):
  self.assertEqual(self.client.get('/health').json()['status'],'ok')
  self.assertEqual(self.client.get('/v1/head/jobs',headers={'X-API-Key':'wrong'}).status_code,401)
 def test_all_non_health_routes_fail_closed_without_configured_key(self):
  from unittest.mock import patch
  app=create_app(self.service,api_key='')
  with patch.dict('os.environ',{'HEADAPI_API_KEY':''}):
   client=TestClient(app)
   self.assertEqual(client.get('/health').status_code,200)
   self.assertEqual(client.get('/').status_code,503)
   self.assertEqual(client.get('/v1/head/jobs/'+'0'*32).status_code,503)
 def test_backend_contract_requires_key_and_front_view(self):
  from uuid import uuid4
  request={'job_id':str(uuid4()),'attempt_id':str(uuid4()),
           'input_images':[{'image_id':str(uuid4()),'input_url':'https://res.cloudinary.com/'+view+'.jpg','view_hint':view}
                           for view in ('left','right','back')]}
  self.assertEqual(self.client.post('/jobs',json=request,headers={'X-API-Key':'wrong'}).status_code,401)
  response=self.client.post('/jobs',json=request,headers={'X-API-Key':'test-backend-key'})
  self.assertEqual(response.status_code,422,response.text)
  self.assertFalse(self.executor.pending)
 def test_remote_input_url_rejects_non_https_and_untrusted_hosts(self):
  from fastapi import HTTPException
  from tools.headapi.api import _fetch_remote_image
  for url in ('http://res.cloudinary.com/a.jpg','https://127.0.0.1/a.jpg',
              'https://res.cloudinary.com.evil.example/a.jpg'):
   with self.subTest(url=url),self.assertRaises(HTTPException):
    _fetch_remote_image(url,{'res.cloudinary.com'})
 def test_backend_status_and_glb_endpoint_match_attempt(self):
  from uuid import uuid4
  job_id=str(uuid4());attempt_id=str(uuid4())
  self.service.submit({view:normalize(image((80+i*40,100,120)))
                       for i,view in enumerate(('front','left','right'))},
                      job_id=job_id,attempt_id=attempt_id)
  folder=self.service.jobs/job_id/'hair';folder.mkdir(parents=True)
  output=folder/'scene.glb';output.write_bytes(b'glTF'+b'\x02\x00\x00\x00'+b'\x04\x00\x00\x00')
  self.service.update(job_id,status='MVP_EXPORTED',artifacts=[{'path':'hair/scene.glb'}])
  headers={'X-API-Key':'test-backend-key'}
  status=self.client.get('/jobs/'+job_id,headers=headers)
  self.assertEqual(status.status_code,200);self.assertEqual(status.json()['status'],'done')
  self.assertEqual(status.json()['attempt_id'],attempt_id)
  self.assertEqual(self.client.get('/jobs/'+job_id+'/outputs/glb',params={'attempt_id':attempt_id},headers=headers).content,output.read_bytes())
  self.assertEqual(self.client.get('/jobs/'+job_id+'/outputs/glb',params={'attempt_id':str(uuid4())},headers=headers).status_code,404)
 def test_assembly_endpoint_is_removed(self):
  self.assertEqual(self.client.post('/v1/head/jobs/'+'0'*32+'/assemble').status_code,404)
 def test_three_views_required_and_back_optional(self):
  self.assertEqual(self.client.post('/v1/head/jobs',files={'front':('front.png',image(),'image/png')}).status_code,422)
  self.assertFalse(self.executor.pending)
  with self.assertRaisesRegex(ValueError,'front, left and right'):
   self.service.submit({'front':normalize(image())})
  self.config['max_pending_jobs']=10;self.service.config['max_pending_jobs']=10
  for slots in [('front','left','right'),('front','left','right','back')]:
   with self.subTest(slots=slots):
    r=self.post(slots);self.assertEqual(r.status_code,202,r.text)
    row=self.service.get(r.json()['job_id']);self.assertEqual([v['slot'] for v in row['observations']],list(slots))
    self.assertEqual(row['rdf_views_actually_used'],[]);self.assertFalse(row['hair_cap_exported'])
 def test_legacy_modes_are_rejected(self):self.assertEqual(self.post(mode='raw').status_code,422)
 def test_identical_view_pixels_rejected(self):
  self.assertEqual(self.client.post('/v1/head/jobs',files={s:(s+'.png',image(),'image/png') for s in ['front','left','right']}).status_code,422)
 def test_invalid_image_rejected(self):self.assertEqual(self.client.post('/v1/head/jobs',files={'front':('bad.jpg',b'not-image','image/jpeg')}).status_code,422)
 def test_queue_capacity(self):
  for _ in range(3):self.assertEqual(self.post().status_code,202)
  self.assertEqual(self.post().status_code,429)
 def test_missing_backend_rejected_without_fake_job(self):
  Path(self.config['paths']['rdf_python']).unlink();self.assertEqual(self.post().status_code,503);self.assertFalse(self.executor.pending)
 def test_original_names_cannot_escape_upload_folder(self):
  files={slot:('../../outside.png',image((100+i*20,110,120)),'image/png') for i,slot in enumerate(('front','left','right'))}
  r=self.client.post('/v1/head/jobs',files=files);self.assertEqual(r.status_code,202)
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
  id=self.post().json()['job_id'];(self.service.jobs/id/'inputs/front.png').write_bytes(b'changed')
  self.service.stage=lambda *a,**k:self.fail('Must reject before running any stage')
  self.service.run(id);row=self.service.get(id);self.assertEqual(row['status'],'FAILED')
  self.assertIn('Input changed',row['errors'][0]['error'])
 def test_worker_failed_stage_does_not_mark_complete(self):
  id=self.post().json()['job_id']
  self.service.stage=lambda *a,**k:False
  self.service.run(id);row=self.service.get(id);self.assertEqual(row['status'],'FAILED');self.assertFalse(row['ready_for_machine'])
 def test_orchestration_runs_rdf_and_hair_cap_without_triposr_or_assembly(self):
  id=self.post().json()['job_id'];folder=self.service.jobs/id;calls=[]
  def stage(job,name,args):
   calls.append((name,args))
   if name=='02_rdf_fit':
    (folder/'rdf').mkdir();write(folder/'rdf/report.json',{'status':'FIT_GATES_PASSED','views_used':['front','left','right']})
   if name=='03_hair_cap':
    (folder/'hair').mkdir();write(folder/'hair/report.json',{'status':'HAIR_CAP_EXPORTED','geometry':{'watertight':True}})
    (folder/'hair/scene.glb').write_bytes(b'glTF'+b'\x02\x00\x00\x00'+b'\x04\x00\x00\x00')
   return True
  self.service.stage=stage;self.service.run(id);row=self.service.get(id)
  self.assertEqual(row['status'],'MVP_EXPORTED');self.assertEqual(row['rdf_views_actually_used'],['front','left','right'])
  self.assertTrue(row['hair_cap_exported']);self.assertFalse(row['ready_for_machine'])
  self.assertEqual(next(a[0] for n,a in calls if n=='02_rdf_fit'),self.config['paths']['rdf_python'])
  self.assertEqual(next(a[0] for n,a in calls if n=='03_hair_cap'),self.config['paths']['haircap_python'])
  self.assertFalse(any('triposr' in n.lower() or 'assemble' in n.lower() for n,_ in calls))
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


class RenderSceneTests(unittest.TestCase):
 def test_render_scene_contains_independent_head_and_hair_nodes(self):
  import trimesh
  from tools.headapi.hair_cap_worker import export_render_scene
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   trimesh.creation.box().export(root/'head.glb')
   trimesh.creation.icosphere(subdivisions=1).export(root/'hair.glb')
   manifest=export_render_scene(root/'head.glb',root/'hair.glb',root/'scene.glb')
   scene=trimesh.load(root/'scene.glb',force='scene')
   self.assertEqual(set(scene.graph.nodes_geometry),{'RDF_Head','Hair_Cap'})
   self.assertFalse(manifest['geometry_union_performed'])
   self.assertEqual(manifest['coordinate_frame'],'RDF native coordinates')

class HeadApiClientTests(unittest.TestCase):
 def test_client_sends_api_key_for_submit_and_poll(self):
  from contextlib import redirect_stdout
  from unittest.mock import patch
  from tools.headapi import client
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for slot in ('front','left','right'):(root/(slot+'.png')).write_bytes(image())
   responses=[
    {'job_id':'job-id','status_url':'/jobs/job-id'},
    {'status':'MVP_EXPORTED','current_stage':'complete'},
   ]
   class Response:
    def __init__(self,value):self.value=value
    def __enter__(self):return self
    def __exit__(self,*args):return None
    def read(self):return json.dumps(self.value).encode()
   requests=[]
   def urlopen(request,timeout):
    requests.append(request);return Response(responses.pop(0))
   with patch('sys.argv',['headapi-client','--input',str(root),'--wait']),\
        patch.dict('os.environ',{'HEADAPI_API_KEY':'unit-test-secret'}),\
        patch('urllib.request.urlopen',side_effect=urlopen),\
        patch('time.sleep'),redirect_stdout(io.StringIO()):
    self.assertEqual(client.main(),0)
   self.assertEqual(len(requests),2)
   self.assertTrue(all(request.get_header('X-api-key')=='unit-test-secret' for request in requests))

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
