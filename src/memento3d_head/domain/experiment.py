"""Typed ports for research-only single-image experiments."""
from pathlib import Path
from typing import Protocol
from memento3d_head.domain.contracts import Inspection, Photo, Settings


class FrontGate(Protocol):
    """Measure a single frontal image; does not pretend to validate a photo set."""
    def execute(self, photo: Photo, output: Path, settings: Settings) -> Inspection:
        """Return evidence and frontal quality issues."""
        ...


class ExperimentStore(Protocol):
    """Snapshot validated normalized bytes and persist metadata."""
    def snapshot_front(self, output: Path) -> dict:
        """Copy the normalized front as immutable worker input and describe it."""
        ...

    def write_json(self, path: Path, value: dict) -> None:
        """Persist a JSON report atomically."""
        ...
