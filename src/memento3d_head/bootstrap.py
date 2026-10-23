"""Composition root: bind the CPU adapter, policy and local execution."""
import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

from memento3d_head import __version__
from memento3d_head.adapters.local_store import LocalStore
from memento3d_head.application.validate_photos import validate_photos
from memento3d_head.domain.contracts import Photo, ValidationError
from memento3d_head.pipelines.photo_set_validation_v1 import PhotoSetValidationV1
from memento3d_head.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    """Expose config-check, doctor and real CPU validation; never emulate reconstruction."""
    parser = argparse.ArgumentParser(prog="memento3d-head")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("config-check", "doctor", "validate-photos"):
        child = commands.add_parser(command)
        child.add_argument("--config", type=Path)
        if command != "config-check":
            child.add_argument("--model", type=Path, required=True)
            child.add_argument("--model-sha256", required=True)
        if command == "validate-photos":
            for slot in ("front", "left", "right"):
                child.add_argument(f"--{slot}", type=Path, required=True)
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--source-kind", choices=("real_capture", "synthetic", "unknown"), required=True)
    from memento3d_head.head_commands import add_commands, execute
    add_commands(commands)
    from memento3d_head.head_optimization_commands import add_command
    add_command(commands)
    args = parser.parse_args(argv)
    if args.command == "optimize-head":
        from memento3d_head.head_optimization_commands import execute as optimize_head
        return optimize_head(args)
    from memento3d_head.research_commands import COMMANDS as research_commands
    if args.command in {"mica-status", "mica-pin", "mica-doctor", "reconstruct-head"} | research_commands:
        return execute(args)
    output = None
    inspector = None
    try:
        settings = load_settings(args.config)
        if args.command == "config-check":
            print(json.dumps({"status": "CONFIG_VALID", "settings": asdict(settings)}, indent=2))
            return 0
        if len(args.model_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.model_sha256):
            raise ValidationError("INVALID_MODEL_HASH", "Expected a 64-character SHA-256")
        if args.command == "validate-photos":
            photos = tuple(Photo(slot, getattr(args, slot).resolve()) for slot in ("front", "left", "right"))
            if not all(photo.path.is_file() for photo in photos):
                raise ValidationError("INPUT_NOT_FOUND", "One or more input files are missing")
            output = args.output.resolve()
            try:
                output.mkdir(parents=True, exist_ok=False)
            except FileExistsError as error:
                raise ValidationError("OUTPUT_EXISTS", "Use a fresh output directory") from error
        from memento3d_head.adapters.mediapipe_inspector import MediaPipeInspector
        inspector = MediaPipeInspector(args.model.resolve(), args.model_sha256)
        provenance = {"package_version": __version__, "backend": "mediapipe", "device": "CPU",
                      "mediapipe_version": importlib.metadata.version("mediapipe"),
                      "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                      "calibrated": False}
        if args.command == "doctor":
            print(json.dumps({"status": "MODEL_LOADED", "provenance": provenance,
                              "inference_tested": False}, indent=2))
            return 0
        provenance["source_kind_declared"] = args.source_kind
        report = validate_photos(photos, output, settings, PhotoSetValidationV1(inspector), LocalStore(), provenance)
        print(json.dumps({"status": report["status"], "report_path": str(output / "report.json"),
                          "error": report["error"]}, ensure_ascii=False, indent=2))
        return {"NEEDS_HUMAN_REVIEW": 0, "NEEDS_RETAKE": 2, "CANCELLED": 130}.get(report["status"], 1)
    except Exception as error:
        failure = {"schema_version": 1, "status": "FAILED", "error": {
            "code": error.code if isinstance(error, ValidationError) else "SETUP_FAILED",
            "message": str(error), "retryable": False}}
        # Never overwrite an existing run on OUTPUT_EXISTS.
        if output is not None and not (output / "report.json").exists() and not isinstance(error, FileExistsError):
            if failure["error"]["code"] != "OUTPUT_EXISTS":
                LocalStore().write_json(output / "report.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1
    finally:
        if inspector is not None:
            inspector.close()
