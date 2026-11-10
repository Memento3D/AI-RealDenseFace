"""Provisional relative regression screen, never a calibrated acceptance gate."""
import math
from memento3d_head.domain.projection_regression import ProjectionPoint, RegressionResult


def screen(baseline: tuple[ProjectionPoint, ...], candidate: tuple[ProjectionPoint, ...]) -> RegressionResult:
    def index(points):
        result = {}
        for point in points:
            key = (point.view, point.name)
            if key in result:
                raise ValueError('Duplicate projection point')
            if not math.isfinite(point.residual) or point.residual < 0:
                raise ValueError('Invalid normalized projection residual')
            result[key] = point
        return result
    original, proposed = index(baseline), index(candidate)
    reasons = []
    compared = 0
    for key, point in original.items():
        # Jaw mapping requires separate contour verification; exclude it here.
        if point.name.startswith('jaw_') or not point.visible or not point.in_frame:
            continue
        other = proposed.get(key)
        if other is None or not other.visible or not other.in_frame:
            reasons.append('/'.join(key) + ': observation coverage lost')
            continue
        compared += 1
        # Engineering trial threshold, chosen to screen relative regressions.
        if other.residual > 1.5 * point.residual + 0.005:
            reasons.append('/'.join(key) + ': residual increased')
    status = 'REGRESSION_FLAGGED' if reasons else ('NO_REGRESSION_FLAGGED' if compared else 'INSUFFICIENT_EVIDENCE')
    return RegressionResult(status, tuple(reasons), compared)
