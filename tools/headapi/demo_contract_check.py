"""In-process HTTP output contract replay using actual scene bytes; NOT web E2E."""
import argparse,json,secrets
from pathlib import Path
from uuid import uuid4
from fastapi.testclient import TestClient
from tools.headapi.api import create_app
from tools.headapi.service import sha
from tools.headapi.demo_scene_worker import inspect_glb


class ReadOnlyReplayFixture:
    def __init__(self,path,expected_sha):
        self.path=path;self.job=str(uuid4());self.attempt=str(uuid4());self.config={'face_render_policy':'strict_uv_v1'}
        if sha(path)!=expected_sha:raise ValueError('Scene artifact changed')
    def start(self):pass
    def close(self):pass
    def preflight(self,mode):
        if mode!='mvp' or not self.path.is_file():raise ValueError('Fixture scene unavailable')
    def get(self,job):
        if job!=self.job:raise KeyError(job)
        return {'job_id':self.job,'attempt_id':self.attempt,'status':'MVP_EXPORTED','scene_artifact':'render/scene.glb','errors':[]}
    def artifact(self,job,relative):
        self.get(job)
        if relative!='render/scene.glb':raise KeyError(relative)
        return self.path


def run(rehearsal,output):
    rehearsal=Path(rehearsal).resolve();report=json.loads((rehearsal/'report.json').read_text());rows=[]
    for case in report['cases']:
        if case.get('scene',{}).get('status')!='COMPLETED':raise ValueError('Replay scene did not complete')
        path=rehearsal/case['case']/'render/scene.glb';fixture=ReadOnlyReplayFixture(path,case['render_report']['scene_sha256']);key=secrets.token_hex(24)
        app=create_app(fixture,api_key=key)
        with TestClient(app) as client:
            base='/jobs/'+fixture.job
            health=client.get('/health');status=client.get(base,headers={'X-API-Key':key})
            good=client.get(base+'/outputs/glb',params={'attempt_id':fixture.attempt},headers={'X-API-Key':key})
            wrong=client.get(base+'/outputs/glb',params={'attempt_id':str(uuid4())},headers={'X-API-Key':key})
            denied=client.get(base+'/outputs/glb',params={'attempt_id':fixture.attempt})
            assert health.status_code==200 and health.json()['face_render_policy']=='strict_uv_v1'
            assert status.status_code==200 and status.json()['status']=='done'
            assert good.status_code==200 and good.content==path.read_bytes() and good.headers['content-type']=='model/gltf-binary'
            assert wrong.status_code==404 and denied.status_code==401
        rows.append({'case':case['case'],'status_http':status.status_code,'download_http':good.status_code,'wrong_attempt_http':wrong.status_code,'unauthenticated_http':denied.status_code,'glb_sha256':sha(path),'inspection':inspect_glb(path)})
    with Path(output).open('x',encoding='utf-8') as file:json.dump({'scope':'in-process API output replay with read-only fixture service and real scenes; no submit, network fetch, new inference, Spring or browser','web_backend_e2e_verified':False,'cases':rows},file,indent=2)
    print('PASS: actual GLB bytes through API route, attempt isolation, authentication and embedded textures')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rehearsal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.rehearsal,a.output)
