"""Synthetic protocol/tamper tests; file fixtures are not inference evidence."""
import json
from pathlib import Path
import unittest
from uuid import uuid4
from tools.headapi.service import sha,write
from tools.headapi.api import _backend_status
import test_headapi as fixtures


class CustomizationAPITests(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.APITests('test_mvp_is_default_upload_mode');self.case.setUp()
        self.client=self.case.client;self.service=self.case.service
        self.job=self.case.post().json()['job_id'];self.attempt=str(uuid4())
        self.root=self.case.root/'library';self.root.mkdir()
        assets=[]
        for aid in ['graduation-cap','hair-side-part','glasses-round']:
            p=self.root/(aid+'.glb');p.write_bytes(b'explicit synthetic file fixture: '+aid.encode())
            thumb=self.root/(aid+'.png');thumb.write_bytes(b'explicit thumbnail fixture')
            assets.append({'id':aid,'version':'v1','name':aid,'kind':'glasses' if aid=='glasses-round' else 'headwear',
                'preview_file':p.name,'preview_sha256':sha(p),'preview_bytes':p.stat().st_size,
                'thumbnail_file':thumb.name,'thumbnail_sha256':sha(thumb),'requires_per_job_fit':True,
                'supported_body_preset':'graduation-standing-base-v1'})
        write(self.root/'catalog.json',{'status':'MVP_SELECTED_PREVIEW','selection':'mvp-v1','assets':assets,
            'groups':[],'default_selection':{'headwear':'graduation-cap','glasses':None}})
        write(self.root/'fitting-profile.json',{'policy':'mvp_landmark_fit_v1'})
        self.service.config['customization']={'policy':'mvp_landmark_fit_v1','library':str(self.root),
            'catalog_sha256':sha(self.root/'catalog.json'),'profile_sha256':sha(self.root/'fitting-profile.json')}
        self.folder=self.service.jobs/self.job
        (self.folder/'render').mkdir();(self.folder/'render/scene.glb').write_bytes(b'explicit base fixture')
        (self.folder/'customization').mkdir()
        entries=[]
        for a in assets:
            entry={'asset_id':a['id'],'sha256':a['preview_sha256'],'geometry_mode':'shared_transform'}
            if a['id']!='graduation-cap':
                p=self.folder/'customization'/(a['id']+'.glb');p.write_bytes(b'explicit fitted fixture')
                entry.update(sha256=sha(p),geometry_mode='fitted_geometry')
            entries.append(entry)
        write(self.folder/'customization/manifest.json',{'job_id':self.job,'attempt_id':self.attempt,
            'base_model':{'sha256':sha(self.folder/'render/scene.glb')},'assets':entries})
        self.service.update(self.job,status='MVP_EXPORTED',attempt_id=self.attempt,output_kind='customization_base',
            customization_ready=True,customization_manifest_artifact='customization/manifest.json',
            customization_manifest_sha256=sha(self.folder/'customization/manifest.json'),artifacts=self.service.collect(self.job))

    def tearDown(self):self.case.tearDown()

    def url(self):return '/jobs/'+self.job+'/customization'

    def test_catalog_and_files_require_auth_and_omit_local_paths(self):
        self.assertEqual(self.client.get('/customization/catalog',headers={'X-API-Key':''}).status_code,401)
        response=self.client.get('/customization/catalog')
        self.assertEqual(response.status_code,200);self.assertEqual(len(response.json()['assets']),3)
        self.assertNotIn(str(self.root),response.text)
        self.assertEqual(self.client.get('/customization/assets/glasses-round/thumbnail').status_code,200)
        self.assertEqual(self.client.get('/customization/assets/unknown/preview').status_code,404)

    def test_scene_and_shared_or_fitted_files_are_bound_to_attempt(self):
        self.assertEqual(self.client.get(self.url(),params={'attempt_id':self.attempt}).status_code,200)
        self.assertEqual(self.client.get(self.url(),params={'attempt_id':str(uuid4())}).status_code,404)
        self.assertEqual(self.client.get(self.url(),params={'attempt_id':'invalid'}).status_code,422)
        for aid in ['graduation-cap','hair-side-part','glasses-round']:
            response=self.client.get(self.url()+'/assets/'+aid,params={'attempt_id':self.attempt})
            self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.get(self.url()+'/assets/unknown',params={'attempt_id':self.attempt}).status_code,404)

    def test_changed_manifest_and_base_are_not_delivered(self):
        (self.folder/'render/scene.glb').write_bytes(b'changed base')
        self.assertEqual(self.client.get(self.url(),params={'attempt_id':self.attempt}).status_code,409)

    def test_changed_library_is_not_delivered(self):
        (self.root/'glasses-round.glb').write_bytes(b'changed shared geometry')
        self.assertEqual(self.client.get('/customization/catalog').status_code,503)
        self.assertEqual(self.client.get(self.url()+'/assets/graduation-cap',params={'attempt_id':self.attempt}).status_code,409)

    def test_changed_fitted_geometry_is_not_delivered(self):
        (self.folder/'customization/glasses-round.glb').write_bytes(b'changed fitted geometry')
        self.assertEqual(self.client.get(self.url()+'/assets/glasses-round',params={'attempt_id':self.attempt}).status_code,409)

    def test_unready_or_failed_job_does_not_expose_customization(self):
        self.service.update(self.job,status='FAILED')
        self.assertEqual(self.client.get(self.url(),params={'attempt_id':self.attempt}).status_code,409)
        result=_backend_status(self.service.get(self.job))
        self.assertFalse(result['customization_ready']);self.assertFalse(result['ready_for_machine'])


if __name__=='__main__':unittest.main()
