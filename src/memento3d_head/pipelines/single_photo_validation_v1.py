"""Single-photo research gate using the existing experimental thresholds."""
from dataclasses import replace
from pathlib import Path
from memento3d_head.domain.contracts import Inspection, Inspector, Photo, Settings, ValidationError


class SinglePhotoValidationV1:
    """Reject bad inputs without asserting multi-view coverage or authenticity."""
    def __init__(self, inspector: Inspector):
        self.inspector = inspector

    def execute(self, photo: Photo, output: Path, settings: Settings) -> Inspection:
        """Apply frontal angle policy to one real inspector result."""
        if photo.slot != 'front':
            raise ValidationError('INVALID_SLOT','Single-image experiments require slot front')
        result=self.inspector.inspect(photo,output,settings)
        issues=list(result.issues)
        if result.face is not None:
            if abs(result.face.yaw)>settings.front_max_yaw:
                issues.append('NOT_FRONTAL')
            if abs(result.face.pitch)>settings.max_pitch:
                issues.append('EXCESSIVE_PITCH')
            if abs(result.face.roll)>settings.max_roll:
                issues.append('EXCESSIVE_ROLL')
        elif 'FACE_COUNT_NOT_ONE' not in issues:
            issues.append('FACE_NOT_MEASURED')
        return replace(result,issues=tuple(issues))
