"""Synthetic lifecycle tests; real GPU/HTTP evidence is stored separately in runs."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.headapi.production_service import ProductionJobService,POLICY
from tools.headapi.service import sha,write

class Held:
    def submit(self,*args):pass
    def shutdown(self,**kwargs):pass

class ProductionServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.runtime=self.root/'runtime';(self.runtime/'GLB').mkdir(parents=True)
        (self.runtime/'GLB/men.glb').write_bytes(b'pinned template fixture')
        (self.runtime/'worker.py').write_text('fixture = True')
        self.python=self.root/'python';self.python.write_bytes(b'fixture')
        self.asset=self.root/'asset';self.asset.write_bytes(b'pinned asset fixture')
        self.mica=self.root/'mica.json';write(self.mica,{'python':str(self.python),'assets':{'fixture':{'path':str(self.asset),'sha256':sha(self.asset)}}})
        self.manifest=self.root/'manifest.json';write(self.manifest,{'files':[{'path':'worker.py','after_byte_sha256':sha(self.runtime/'worker.py')}]})
        self.config={'root':str(self.root),'pipeline_policy':POLICY,'template_sha256':sha(self.runtime/'GLB/men.glb'),'paths':{'runtime':str(self.runtime),'cpu_python':str(self.python),'blender':str(self.python),'mica_config':str(self.mica),'runtime_manifest':str(self.manifest),'jobs':str(self.root/'jobs')}}
        self.service=ProductionJobService(self.config,Held())
    def tearDown(self):self.service.close();self.temp.cleanup()
    def test_asset_mutation_fails_preflight(self):
        self.service.preflight('mvp');self.asset.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'asset pin'):self.service.preflight('mvp')
    def test_runtime_mutation_fails_preflight(self):
        (self.runtime/'worker.py').write_text('changed = True')
        with self.assertRaisesRegex(ValueError,'Runtime pin'):self.service.preflight('mvp')
    def test_worker_failure_never_publishes_success(self):
        images={s:{'png':s.encode(),'original':s.encode(),'format':'PNG','size':[80,100]} for s in ('front','left','right')}
        row=self.service.submit(images);job=row['job_id']
        with patch.object(self.service,'stage',return_value=False):self.service.run(job)
        result=self.service.get(job)
        self.assertEqual(result['status'],'FAILED');self.assertNotIn('scene_artifact',result)
        self.assertFalse(result['ready_for_machine'])
        self.assertTrue(all(not x['used_for_hair_cap'] for x in result['observations']))
    def test_changed_submission_code_rejected_before_worker(self):
        images={s:{'png':s.encode(),'original':s.encode(),'format':'PNG','size':[80,100]} for s in ('front','left','right')}
        row=self.service.submit(images)
        with patch.object(self.service,'code_hashes',return_value={'changed':'yes'}),patch.object(self.service,'stage') as stage:
            self.service.run(row['job_id']);stage.assert_not_called()
        self.assertEqual(self.service.get(row['job_id'])['status'],'FAILED')

if __name__=='__main__':unittest.main()
