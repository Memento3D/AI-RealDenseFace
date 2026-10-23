"""CPU MediaPipe Tasks adapter; no automatic model download or GPU loading."""
import hashlib
import math
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from memento3d_head.domain.contracts import FaceObservation, Inspection, Photo, Settings, ValidationError


def rotation_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Extract approximate yaw/pitch/roll (Rz Ry Rx); sign requires capture calibration."""
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValidationError("INVALID_MODEL_RESULT", "Invalid rotation matrix")
    # Remove scale/shear before extracting Euler angles.
    u, _, vt = np.linalg.svd(rotation)
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    yaw = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return tuple(math.degrees(value) for value in (yaw, pitch, roll))


class MediaPipeInspector:
    """Own a single CPU landmarker; process the three photographs sequentially."""
    def __init__(self, model: Path, expected_sha256: str):
        actual = hashlib.sha256(model.read_bytes()).hexdigest()
        if actual != expected_sha256.lower():
            raise ValidationError("MODEL_CHECKSUM_MISMATCH", "Model checksum does not match")
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        """Release native task resources after success or failure."""
        self.landmarker.close()

    def inspect(self, photo: Photo, destination: Path, settings: Settings) -> Inspection:
        """Decode immutable input bytes, normalize EXIF and save diagnostic artifacts."""
        if photo.path.stat().st_size > settings.max_file_bytes:
            raise ValidationError("FILE_TOO_LARGE", f"{photo.slot}: file exceeds byte limit")
        data = photo.path.read_bytes()
        if len(data) > settings.max_file_bytes:
            raise ValidationError("FILE_TOO_LARGE", f"{photo.slot}: file exceeds byte limit")
        source_hash = hashlib.sha256(data).hexdigest()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as original:
                    if original.format not in {"JPEG", "PNG"} or getattr(original, "n_frames", 1) != 1:
                        raise ValidationError("INVALID_IMAGE", "Only still JPEG/PNG are supported")
                    if original.width * original.height > settings.max_pixels:
                        raise ValidationError("IMAGE_TOO_LARGE", "Pixel limit exceeded")
                    image = ImageOps.exif_transpose(original).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
            raise ValidationError("INVALID_IMAGE", f"{photo.slot}: cannot decode image") from error
        issues = []
        if min(image.size) < settings.min_short_side:
            issues.append("IMAGE_TOO_SMALL")
        preview = image.copy()
        preview.thumbnail((1536, 1536))
        array = np.asarray(preview)
        result = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=array))
        face = None
        sharpness = brightness = None
        if len(result.face_landmarks) != 1:
            issues.append("FACE_COUNT_NOT_ONE")
        elif len(result.facial_transformation_matrixes) != 1:
            raise ValidationError("INVALID_MODEL_RESULT", "Face rotation is missing")
        else:
            landmarks = tuple((float(p.x), float(p.y), float(p.z)) for p in result.face_landmarks[0])
            points = np.asarray(landmarks)
            if not np.isfinite(points).all():
                raise ValidationError("INVALID_MODEL_RESULT", "Non-finite landmarks")
            x0, y0 = points[:, :2].min(axis=0)
            x1, y1 = points[:, :2].max(axis=0)
            bbox = (float(x0), float(y0), float(x1), float(y1))
            yaw, pitch, roll = rotation_angles(result.facial_transformation_matrixes[0])
            face = FaceObservation(bbox, yaw, pitch, roll, landmarks)
            if x0 <= 0.01 or y0 <= 0.01 or x1 >= 0.99 or y1 >= 0.99:
                issues.append("FACE_NEAR_BORDER")
            if (x1-x0) * (y1-y0) < settings.min_face_fraction:
                issues.append("FACE_TOO_SMALL_IN_FRAME")
            if min((x1-x0)*image.width, (y1-y0)*image.height) < settings.min_face_pixels:
                issues.append("FACE_TOO_FEW_PIXELS")
            # Fixed analysis scale makes sharpness comparisons less resolution-dependent.
            box = (max(0, int(x0*image.width)), max(0, int(y0*image.height)),
                   min(image.width, math.ceil(x1*image.width)), min(image.height, math.ceil(y1*image.height)))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValidationError("INVALID_MODEL_RESULT", "Empty face bounds")
            gray = np.asarray(image.crop(box).convert("L").resize((256, 256)), dtype=np.float32)
            lap = (-4*gray[1:-1,1:-1] + gray[:-2,1:-1] + gray[2:,1:-1]
                   + gray[1:-1,:-2] + gray[1:-1,2:])
            sharpness, brightness = float(lap.var()), float(gray.mean())
            if sharpness < settings.min_sharpness:
                issues.append("LOW_SHARPNESS")
            if not settings.min_brightness <= brightness <= settings.max_brightness:
                issues.append("EXPOSURE_OUT_OF_RANGE")
            draw = ImageDraw.Draw(preview)
            for x, y, _ in landmarks:
                cx, cy = x*preview.width, y*preview.height
                draw.ellipse((cx-1, cy-1, cx+1, cy+1), fill="lime")
            draw.text((10, 10), f"yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}", fill="red")
        image.save(destination / f"{photo.slot}.normalized.png")
        preview.save(destination / f"{photo.slot}.landmarks.jpg")
        return Inspection(photo.slot, source_hash, image.width, image.height,
                          sharpness, brightness, face, tuple(issues))
