"""Typed diagnostic observations; no manufacturing or likeness acceptance."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionPoint:
    view: str
    name: str
    residual: float
    visible: bool
    in_frame: bool


@dataclass(frozen=True)
class RegressionResult:
    status: str
    reasons: tuple[str, ...]
    compared_points: int
