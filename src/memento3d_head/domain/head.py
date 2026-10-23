"""Ports for a measured, single-front-image head initialization attempt."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class HeadRequest:
    """Bind original source bytes and human review to a validation report."""
    validation_run: Path
    front: Path
    reviewed_by: str
    allow_synthetic: bool = False


class HeadBackend(Protocol):
    """Execute geometry inference in an isolated model environment."""
    def reconstruct(self, front: Path, output: Path) -> dict:
        """Return real artifacts and measured model/runtime provenance."""
        ...


class HeadStore(Protocol):
    """Verify the approved source, normalize a snapshot and describe outputs."""
    def prepare(self, request: HeadRequest, output: Path) -> dict:
        """Verify report/source association before supplying any model input."""
        ...

    def write_json(self, path: Path, value: dict) -> None:
        """Persist report atomically."""
        ...
