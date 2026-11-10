"""Durable local Production adapter; model SDKs execute only in subprocesses."""
import json
from datetime import datetime,timezone
from pathlib import Path
from .service import JobService, sha

POLICY='production_direct_bounded_v1'

class ProductionJobService(JobService):
    def preflight(self,mode):
        if mode!='mvp' or self.config.get('pipeline_policy')!=POLICY:raise ValueError('Unsupported Production policy')
        for name in ('cpu_python','runtime','blender','mica_config','runtime_manifest'):
            if not Path(self.config['paths'][name]).exists():raise ValueError('Missing prerequisite: '+name)
        runtime=Path(self.config['paths']['runtime'])
        manifest=json.loads(Path(self.config['paths']['runtime_manifest']).read_text(encoding='utf-8'))
        for item in manifest['files']:
            file=runtime/item['path']
            if not file.is_file() or sha(file)!=item['after_byte_sha256']:raise ValueError('Runtime pin changed: '+item['path'])
        mica=json.loads(Path(self.config['paths']['mica_config']).read_text(encoding='utf-8'))
        if not Path(mica['python']).is_file():raise ValueError('MICA environment missing')
        for item in mica['assets'].values():
            if not Path(item['path']).is_file() or sha(item['path'])!=item['sha256']:raise ValueError('MICA asset pin changed')
        if not (runtime/'GLB/men.glb').is_file() or sha(runtime/'GLB/men.glb')!=self.config['template_sha256']:raise ValueError('Template pin changed')
    def code_hashes(self):
        pins=super().code_hashes()
        runtime=Path(self.config['paths']['runtime'])
        for p in sorted(runtime.rglob('*.py')):
            if '.venv' not in p.parts:pins['production-runtime/'+p.relative_to(runtime).as_posix()]=sha(p)
        pins['production-mica-config']=sha(self.config['paths']['mica_config'])
        pins['production-runtime-manifest']=sha(self.config['paths']['runtime_manifest'])
        return pins
    def run(self,job_id):
        folder=self.jobs/job_id
        self.update(job_id,status='RUNNING',pipeline_policy=POLICY,quality_status='NEEDS_VISUAL_REVIEW',display_only=True)
        try:
            row=self.get(job_id)
            observations=row['observations']
            for entry in observations:entry.update(requested_for_rdf=False,used_for_hair_cap=False)
            self.update(job_id,observations=observations,started_at=datetime.now(timezone.utc).isoformat())
            if self.code_hashes()!=row['code_sha256']:raise ValueError('Code changed after submission')
            self.preflight('mvp')
            for entry in row['observations']:
                if sha(folder/'inputs'/(entry['slot']+'.png'))!=entry['normalized_sha256']:raise ValueError('Input changed after submission')
            if not self.stage(job_id,'production_reconstruction',[self.config['paths']['cpu_python'],'-m','tools.headapi.production_worker','--job',folder]):raise ValueError('Production worker failed; inspect stage log')
            report=json.loads((folder/'production-report.json').read_text(encoding='utf-8'))
            if report['status']!='EXPORTED_NEEDS_VISUAL_REVIEW' or sha(folder/'render/scene.glb')!=report['web_sha256']:raise ValueError('Output publication check failed')
            self.update(job_id,status='MVP_EXPORTED',scene_artifact='render/scene.glb',artifacts=self.collect(job_id),ready_for_machine=False,finished_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:self.update(job_id,status='FAILED',error=str(exc),artifacts=self.collect(job_id),ready_for_machine=False,finished_at=datetime.now(timezone.utc).isoformat())
