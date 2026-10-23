"""Research-only single-image initialization; never bypasses a three-view report."""
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from memento3d_head.domain.contracts import Photo, Settings, ValidationError
from memento3d_head.domain.experiment import ExperimentStore, FrontGate
from memento3d_head.domain.head import HeadBackend


def experiment_head(photo: Photo, output: Path, settings: Settings, source_kind: str,
                    gate: FrontGate, backend: HeadBackend, store: ExperimentStore,
                    validator_provenance: dict) -> dict:
    """Validate a standalone input, then execute real inference or retain rejection."""
    started=time.monotonic()
    report={'schema_version':1,'attempt_id':str(uuid4()),
            'pipeline_type':'single_image_experiment','pipeline_version':1,
            'started_at':datetime.now(timezone.utc).isoformat(),
            'research_only':True,'status':'FAILED','error':None,'artifacts':{},
            'multi_view_fitting':False,'has_texture':False,
            'likeness_verified':False,'printability_verified':False,
            'identity_verified':False,'real_capture_verified':False,
            'settings':asdict(settings),'validator_provenance':validator_provenance,
            'input':{'source_kind_declared':source_kind,'views_used_for_geometry':['front'],
                     'photo_set_validated':False,'human_review_completed':False}}
    try:
        inspection=gate.execute(photo,output,settings)
        report['quality']=asdict(inspection)
        report['input']['source_sha256']=inspection.source_sha256
        if inspection.issues:
            report['status']='NEEDS_RETAKE'
        else:
            report['input'].update(store.snapshot_front(output))
            report.update(backend.reconstruct(output/'front.input.png',output))
            report['status']='GEOMETRY_EXPORTED'
    except KeyboardInterrupt:
        report['status']='CANCELLED'
        report['error']={'code':'INTERRUPTED','message':'Single-image experiment interrupted'}
    except Exception as error:
        report['error']={'code':error.code if isinstance(error,ValidationError) else 'EXPERIMENT_FAILED',
                         'message':str(error),'retryable':False}
    finally:
        report['elapsed_seconds']=round(time.monotonic()-started,3)
        store.write_json(output/'report.json',report)
    return report
