"""Orchestrate one head initialization without model or image SDK imports."""
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from memento3d_head.domain.contracts import ValidationError
from memento3d_head.domain.head import HeadBackend, HeadRequest, HeadStore


def reconstruct_head(request: HeadRequest, output: Path, backend: HeadBackend, store: HeadStore) -> dict:
    """Retain evidence on success, invalid input, model failure or cancellation."""
    started = time.monotonic()
    report = {"schema_version": 1, "attempt_id": str(uuid4()),
              "pipeline_type": "head_initialization", "pipeline_version": 1,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "FAILED", "error": None, "artifacts": {},
              "multi_view_fitting": False, "has_texture": False,
              "printability_verified": False, "likeness_verified": False}
    try:
        report["input"] = store.prepare(request, output)
        result = backend.reconstruct(output / "front.input.png", output)
        report.update(result)
        report["status"] = "GEOMETRY_EXPORTED"
    except KeyboardInterrupt:
        report["status"] = "CANCELLED"
        report["error"] = {"code": "INTERRUPTED", "message": "Worker interrupted"}
    except Exception as error:
        report["error"] = {"code": error.code if isinstance(error, ValidationError) else "RECONSTRUCTION_FAILED",
                           "message": str(error), "retryable": False}
    finally:
        report["elapsed_seconds"] = round(time.monotonic()-started, 3)
        store.write_json(output / "report.json", report)
    return report
