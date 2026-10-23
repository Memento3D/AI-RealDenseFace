"""Subprocess execution and verification of real head artifacts."""
import json
import os
import signal
import subprocess
from pathlib import Path

from memento3d_head.adapters.head_store import sha256
from memento3d_head.adapters.mica_assets import verify_config
from memento3d_head.domain.contracts import ValidationError


def stop_worker(process):
    """Stop the process group before propagating timeout or user cancellation."""
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
    else:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_worker(config_path: Path, output: Path, front: Path | None = None) -> dict:
    """Execute an external interpreter with a bounded deadline and retained log."""
    config=verify_config(config_path)
    command=[config['python'],str(Path(__file__).with_name('mica_worker.py').resolve()),
             '--config',str(config_path.resolve()),'--output',str(output.resolve())]
    if front is not None:
        command.extend(['--input',str(front.resolve())])
    flags={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
    with (output/'mica_worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=config['repo'],stdout=log,stderr=subprocess.STDOUT,**flags)
        try:
            code=process.wait(timeout=config['timeout_seconds'])
        except subprocess.TimeoutExpired as error:
            stop_worker(process)
            raise ValidationError('WORKER_TIMEOUT','See mica_worker.log') from error
        except KeyboardInterrupt:
            stop_worker(process)
            raise
    if code!=0:
        raise ValidationError('MICA_WORKER_FAILED','See mica_worker.log and worker_result.json; no automatic fallback')
    result=json.loads((output/'worker_result.json').read_text(encoding='utf-8'))
    expected='MODEL_LOADED' if front is None else 'GEOMETRY_EXPORTED'
    if result.get('status')!=expected:
        raise ValidationError('INVALID_WORKER_RESULT','Worker did not produce the required result')
    return result


class MicaBackend:
    """Use the pinned external pretrained model; no placeholder inference."""
    def __init__(self, config: Path, *, smooth_preview: bool = False):
        self.config=config
        self.smooth_preview=smooth_preview

    def reconstruct(self, front: Path, output: Path) -> dict:
        """Validate real OBJ geometry and generate a simple inspection preview."""
        result=run_worker(self.config,output,front)
        from memento3d_head.adapters.mesh_preview import validate_and_preview
        measurements=validate_and_preview(output/'head.obj',output/'head.preview.png')
        artifacts={}
        for name in ('head.obj','head.ply','identity.npy','front.aligned.jpg','head.preview.png'):
            file=output/name
            if not file.is_file() or file.stat().st_size==0:
                raise ValidationError('MISSING_ARTIFACT',name)
            artifacts[name]={'path':name,'sha256':sha256(file),'size_bytes':file.stat().st_size}
        if self.smooth_preview:
            from memento3d_head.adapters.smooth_preview import render_smooth
            measurements['smooth_preview']=render_smooth(output/'head.obj',output/'head.preview.smooth.png')
            preview=output/'head.preview.smooth.png'
            artifacts[preview.name]={'path':preview.name,'sha256':sha256(preview),'size_bytes':preview.stat().st_size}
        return {'provenance':result,'geometry':measurements,'artifacts':artifacts}
