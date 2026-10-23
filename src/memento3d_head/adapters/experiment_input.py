"""Short-lived CPU inspection and normalized input snapshots."""
from pathlib import Path
from memento3d_head.adapters.head_store import LocalHeadStore, sha256
from memento3d_head.domain.contracts import Inspection, Photo, Settings


class OneShotMediaPipeInspector:
    """Load the CPU validator only during inspection; close it before GPU work."""
    def __init__(self, model: Path, expected_sha256: str):
        self.model=model
        self.expected_sha256=expected_sha256

    def inspect(self, photo: Photo, destination: Path, settings: Settings) -> Inspection:
        """Measure one image and release the task even on failure."""
        from memento3d_head.adapters.mediapipe_inspector import MediaPipeInspector
        inspector=MediaPipeInspector(self.model,self.expected_sha256)
        try:
            result=inspector.inspect(photo,destination,settings)
        except BaseException:
            try:
                inspector.close()
            except Exception:
                pass
            raise
        inspector.close()
        return result


class LocalExperimentStore(LocalHeadStore):
    """Bind inference to the exact normalized image measured by the CPU gate."""
    def snapshot_front(self, output: Path) -> dict:
        """Copy without altering pixels and record normalized input SHA256."""
        original=output/'front.normalized.png'
        destination=output/'front.input.png'
        with original.open('rb') as source, destination.open('xb') as target:
            import shutil
            shutil.copyfileobj(source,target)
        if sha256(original)!=sha256(destination):
            raise OSError('Normalized input snapshot checksum mismatch')
        return {'normalized_sha256':sha256(destination),'worker_input':'front.input.png'}
