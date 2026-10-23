"""Application orchestration depends only on domain ports."""
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from memento3d_head.domain.contracts import Photo, Pipeline, Settings, Store, ValidationError


def validate_photos(photos: tuple[Photo, ...], output: Path, settings: Settings,
                    pipeline: Pipeline, store: Store, provenance: dict) -> dict:
    """Execute one attempt and retain a report on failure, rejection or interruption."""
    start = time.monotonic()
    report = {
        "schema_version": 1, "attempt_id": str(uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_type": settings.pipeline_type, "pipeline_version": settings.pipeline_version,
        "settings": asdict(settings), "provenance": provenance, "status": "FAILED",
        "error": None, "assessment": None,
    }
    try:
        report["assessment"] = asdict(pipeline.execute(photos, output, settings))
        report["status"] = report["assessment"]["status"]
    except KeyboardInterrupt:
        report["status"] = "CANCELLED"
        report["error"] = {"code": "INTERRUPTED", "message": "Validation interrupted"}
    except Exception as error:
        report["error"] = {"code": error.code if isinstance(error, ValidationError) else "VALIDATION_FAILED",
                           "message": str(error), "retryable": False}
    finally:
        report["elapsed_seconds"] = round(time.monotonic()-start, 3)
        store.write_json(output / "report.json", report)
    return report
