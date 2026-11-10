"""Contract orchestration doubles and real synthetic-mesh composition; no inference."""
import json,sys,tempfile,unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
import test_headapi as fixtures
from tools.headapi.service import write,sha


class UVDemoAPITests(unittest.TestCase):
    setUp=fixtures.APITests.setUp
    tearDown=fixtures.APITests.tearDown
    post=fixtures.APITests.post

    def stages(self,job_id,fail=None):
        folder=self.service.jobs/job_id;calls=[]
        def stage(_job,name,args):
            calls.append(name)
            if name==fail:return False
            if name=='02_rdf_fit':
                (folder/'rdf').mkdir();write(folder/'rdf/report.json',{'status':'FIT_GATES_PASSED','views_used':['front','left','right']})
            if name=='03b_uv_texture':
                path=folder/'rdf_uv/final/face_textured.glb';path.parent.mkdir(parents=True);path.write_bytes(b'explicit UV contract fixture')
                write(folder/'rdf_uv/report.json',{'status':'UV_FACE_EXPORTED','head_sha256':sha(path)})
            if name=='03_hair_cap':
                (folder/'hair').mkdir();(folder/'hair/scene.glb').write_bytes(b'glTF'+bytes(12))
                write(folder/'hair/report.json',{'status':'HAIR_CAP_EXPORTED','geometry':{'watertight':True}})
            if name=='04_scene_export':
                (folder/'render').mkdir();path=folder/'render/scene.glb';path.write_bytes(b'glTF'+b'UV scene contract fixture')
                write(folder/'render/report.json',{'status':'UV_SCENE_EXPORTED','scene_sha256':sha(path),'presentation_policy':self.service.config.get('presentation_policy','none')})
            return True
        self.service.stage=stage
        return calls

    def test_opt_in_runs_uv_and_composition_and_downloads_new_artifact(self):
        self.service.config['face_render_policy']='strict_uv_v1'
        job=self.post().json()['job_id'];calls=self.stages(job);self.service.run(job)
        row=self.service.get(job);self.assertEqual(row['status'],'MVP_EXPORTED');self.assertEqual(row['scene_artifact'],'render/scene.glb')
        self.assertEqual(calls,['01_rdf_observations','02_rdf_fit','02b_rdf_export','03_hair_cap','03b_uv_texture','04_scene_export'])
        attempt=str(uuid4());self.service.update(job,attempt_id=attempt)
        result=self.client.get('/jobs/'+job+'/outputs/glb',params={'attempt_id':attempt})
        self.assertEqual(result.status_code,200);self.assertEqual(result.content,(self.service.jobs/job/'render/scene.glb').read_bytes())
        self.assertFalse(row['ready_for_machine'])

    def test_uv_failure_never_silently_returns_old_scene(self):
        self.service.config['face_render_policy']='strict_uv_v1';job=self.post().json()['job_id']
        calls=self.stages(job,'03b_uv_texture');self.service.run(job)
        self.assertEqual(self.service.get(job)['status'],'PARTIAL');self.assertNotIn('04_scene_export',calls)
        self.assertEqual(self.client.get('/jobs/'+job+'/outputs/glb',params={'attempt_id':str(uuid4())}).status_code,409)

    def test_scene_failure_is_not_done(self):
        self.service.config['face_render_policy']='strict_uv_v1';job=self.post().json()['job_id']
        self.stages(job,'04_scene_export');self.service.run(job)
        self.assertEqual(self.service.get(job)['status'],'PARTIAL')
        self.assertEqual(self.client.get('/jobs/'+job).json()['status'],'failed')

    def test_unknown_policy_is_rejected(self):
        self.service.config['face_render_policy']='unimplemented'
        self.assertEqual(self.post().status_code,503);self.assertFalse(self.executor.pending)

    def test_uv_progress_maps_to_backend_texturing(self):
        job=self.post().json()['job_id'];self.service.update(job,status='RUNNING',current_stage='03b_uv_texture')
        self.assertEqual(self.client.get('/jobs/'+job).json()['stage'],'texturing')

    def test_done_already_has_download_artifact_before_review_is_written(self):
        self.service.config['face_render_policy']='strict_uv_v1';job=self.post().json()['job_id']
        self.stages(job);observed=[]
        def review(_job):
            row=self.service.get(job);observed.append(row)
            self.assertEqual(row['status'],'MVP_EXPORTED')
            self.assertIn('render/scene.glb',{item['path'] for item in row['artifacts']})
        self.service.review_page=review;self.service.run(job)
        self.assertEqual(len(observed),1)

    def test_presentation_flag_is_bound_to_scene_worker(self):
        self.service.config.update(face_render_policy='strict_uv_v1',presentation_policy='harmonized_head_v1')
        job=self.post().json()['job_id'];self.stages(job);stage=self.service.stage;commands=[]
        def record(job_id,name,args):
            commands.append((name,[str(x) for x in args]));return stage(job_id,name,args)
        self.service.stage=record;self.service.run(job)
        args=next(args for name,args in commands if name=='04_scene_export')
        self.assertEqual(args[args.index('--presentation-policy')+1],'harmonized_head_v1')
        self.assertEqual(self.service.get(job)['status'],'MVP_EXPORTED')

    def test_presentation_cannot_be_enabled_without_uv(self):
        self.service.config['presentation_policy']='harmonized_head_v1'
        self.assertEqual(self.post().status_code,503)


class BudgetTests(unittest.TestCase):
    def test_low_memory_refuses_to_start_child(self):
        from tools.headapi.runtime_budget import run_guarded
        with patch('tools.headapi.runtime_budget.available_ram',return_value=1),patch('tools.headapi.runtime_budget.subprocess.Popen') as popen:
            with self.assertRaises(MemoryError):run_guarded([sys.executable],cwd='.',env=None,stdout=None,timeout=1,minimum=2,abort=1)
            popen.assert_not_called()

    def test_memory_drop_terminates_owned_child(self):
        from tools.headapi.runtime_budget import run_guarded
        with tempfile.TemporaryFile() as log,patch('tools.headapi.runtime_budget.available_ram',side_effect=[3,0]):
            with self.assertRaises(MemoryError):run_guarded([sys.executable,'-c','import time; time.sleep(10)'],cwd='.',env=None,stdout=log,timeout=20,minimum=2,abort=1)


class SyntheticUVSceneTests(unittest.TestCase):
    def fixture(self,root):
        import numpy as np
        import trimesh
        from PIL import Image
        (root/'rdf_uv/final').mkdir(parents=True);(root/'rdf').mkdir();(root/'hair').mkdir()
        mesh=trimesh.creation.box();np.savez(root/'rdf/projection_bundle.npz',vertices=mesh.vertices,faces=mesh.faces)
        mesh.visual=trimesh.visual.texture.TextureVisuals(uv=np.zeros((len(mesh.vertices),2)),image=Image.new('RGB',(8,8),(160,120,100)))
        head=root/'rdf_uv/final/face_textured.glb';mesh.export(head)
        hair=root/'hair/hair_cap_native.glb';trimesh.creation.icosphere(subdivisions=0).export(hair)
        write(root/'rdf_uv/report.json',{'status':'UV_FACE_EXPORTED','head_sha256':sha(head)})
        write(root/'hair/report.json',{'status':'HAIR_CAP_EXPORTED','rdf_bundle_sha256':sha(root/'rdf/projection_bundle.npz'),'output_sha256':{'hair_cap_native.glb':sha(hair)}})

    def test_composition_preserves_real_embedded_texture_and_geometry(self):
        from tools.headapi.demo_scene_worker import run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);report=run(root,root/'render')
            self.assertEqual(report['status'],'UV_SCENE_EXPORTED');self.assertTrue(report['scene_inspection']['embedded_base_color_texture'])
            self.assertTrue(report['head_geometry_preserved']);self.assertFalse(report['geometry_union_performed'])

    def test_cap_from_another_head_is_rejected(self):
        from tools.headapi.demo_scene_worker import run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);cap=json.loads((root/'hair/report.json').read_text());cap['rdf_bundle_sha256']='wrong';write(root/'hair/report.json',cap)
            with self.assertRaisesRegex(ValueError,'another head'):run(root,root/'render')
            self.assertEqual(json.loads((root/'render/report.json').read_text())['status'],'FAILED')
