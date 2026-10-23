"""Three-view policy; numerical thresholds must be calibrated on real captures."""
from pathlib import Path
from memento3d_head.domain.contracts import Assessment, Inspector, Photo, Settings, ValidationError


class PhotoSetValidationV1:
    """Validate angle coverage without claiming identity or source authenticity."""
    def __init__(self, inspector: Inspector):
        self.inspector = inspector

    def execute(self, photos: tuple[Photo, ...], output: Path, settings: Settings) -> Assessment:
        """Inspect sequentially and require opposite signed side rotations."""
        if len(photos) != 3 or {photo.slot for photo in photos} != {"front", "left", "right"}:
            raise ValidationError("INVALID_PHOTO_SET", "Exactly front, left and right are required")
        result = Assessment(status="NEEDS_RETAKE")
        for photo in photos:
            result.photos.append(self.inspector.inspect(photo, output, settings))
        for item in result.photos:
            result.issues.extend(f"{item.slot}:{issue}" for issue in item.issues)
            if item.face is None:
                continue
            face = item.face
            yaw = abs(face.yaw)
            if item.slot == "front" and yaw > settings.front_max_yaw:
                result.issues.append("front:NOT_FRONTAL")
            if item.slot != "front" and not settings.side_min_yaw <= yaw <= settings.side_max_yaw:
                result.issues.append(f"{item.slot}:SIDE_ANGLE_OUT_OF_RANGE")
            if abs(face.pitch) > settings.max_pitch:
                result.issues.append(f"{item.slot}:EXCESSIVE_PITCH")
            if abs(face.roll) > settings.max_roll:
                result.issues.append(f"{item.slot}:EXCESSIVE_ROLL")
        hashes = [item.source_sha256 for item in result.photos]
        if len(set(hashes)) != len(hashes):
            result.issues.append("DUPLICATE_IMAGE_BYTES")
        sides = [item.face for item in result.photos if item.slot != "front"]
        if all(face is not None for face in sides) and len(sides) == 2:
            if sides[0].yaw * sides[1].yaw >= 0:
                result.issues.append("SIDE_VIEWS_NOT_OPPOSITE")
        if not result.issues:
            result.status = "NEEDS_HUMAN_REVIEW"
        return result
