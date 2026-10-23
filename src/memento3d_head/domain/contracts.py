"""Typed contracts, independent of image and model libraries."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class ValidationError(Exception):
    """Expose a stable failure code without retrying automatically."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Settings:
    """Experimental thresholds; not a calibrated production acceptance policy."""
    schema_version: int = 1
    pipeline_type: str = "photo_set_validation"
    pipeline_version: int = 1
    min_short_side: int = 640
    max_file_bytes: int = 20_000_000
    max_pixels: int = 24_000_000
    min_face_fraction: float = 0.12
    min_face_pixels: int = 220
    min_sharpness: float = 35.0
    min_brightness: float = 45.0
    max_brightness: float = 220.0
    front_max_yaw: float = 12.0
    side_min_yaw: float = 25.0
    side_max_yaw: float = 55.0
    max_pitch: float = 15.0
    max_roll: float = 12.0


@dataclass(frozen=True)
class Photo:
    """One explicit capture slot; left/right refers to subject turning direction."""
    slot: str
    path: Path


@dataclass(frozen=True)
class FaceObservation:
    """Normalized landmarks with head rotation estimates in degrees."""
    bbox: tuple[float, float, float, float]
    yaw: float
    pitch: float
    roll: float
    landmarks: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class Inspection:
    """Measured photo evidence; source hash binds results to original bytes."""
    slot: str
    source_sha256: str
    oriented_width: int
    oriented_height: int
    sharpness: float | None
    brightness: float | None
    face: FaceObservation | None
    issues: tuple[str, ...]


@dataclass
class Assessment:
    """Automatic checks cannot verify identity or real photographic origin."""
    status: str
    photos: list[Inspection] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    identity_verified: bool = False
    real_capture_verified: bool = False
    reconstruction_ready: bool = False


class Inspector(Protocol):
    """Read, normalize and analyze images behind the adapter boundary."""
    def inspect(self, photo: Photo, destination: Path, settings: Settings) -> Inspection:
        """Persist a normalized image and diagnostic overlay in the attempt."""
        ...


class Pipeline(Protocol):
    """Execute an explicit validation version."""
    def execute(self, photos: tuple[Photo, ...], output: Path, settings: Settings) -> Assessment:
        """Return measured evidence and a conservative decision."""
        ...


class Store(Protocol):
    """Persist report atomically at the output boundary."""
    def write_json(self, path: Path, value: dict) -> None:
        """Write a JSON report without partial replacement."""
        ...
