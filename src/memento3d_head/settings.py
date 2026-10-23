"""Strict config loading at the composition root."""
import json
import math
from dataclasses import asdict
from pathlib import Path
from memento3d_head.domain.contracts import Settings, ValidationError


def load_settings(path: Path | None) -> Settings:
    """Reject unknown fields, invalid numerical ranges and unsupported versions."""
    raw = json.loads(path.read_text(encoding="utf-8")) if path else {}
    defaults = asdict(Settings())
    if not isinstance(raw, dict) or set(raw)-set(defaults):
        raise ValidationError("INVALID_CONFIG", "Unknown config fields or non-object config")
    for key, value in raw.items():
        expected = defaults[key]
        if isinstance(expected, float):
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = type(value) is type(expected)
        if not valid:
            raise ValidationError("INVALID_CONFIG", f"Invalid type: {key}")
    settings = Settings(**raw)
    if (settings.schema_version, settings.pipeline_type, settings.pipeline_version) != (1, "photo_set_validation", 1):
        raise ValidationError("UNSUPPORTED_VERSION", "Only photo_set_validation v1 is implemented")
    numbers = [value for key, value in asdict(settings).items() if key not in {"pipeline_type"}]
    if any(value <= 0 for value in numbers):
        raise ValidationError("INVALID_CONFIG", "Thresholds must be positive")
    if not (settings.front_max_yaw < settings.side_min_yaw < settings.side_max_yaw < 90
            and settings.max_pitch < 90 and settings.max_roll < 90
            and settings.min_brightness < settings.max_brightness <= 255
            and settings.min_face_fraction < 1):
        raise ValidationError("INVALID_CONFIG", "Invalid threshold ordering or range")
    return settings
