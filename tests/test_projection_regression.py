import unittest
from memento3d_head.domain.projection_regression import ProjectionPoint
from memento3d_head.pipelines.projection_regression_v1 import screen


class RegressionTests(unittest.TestCase):
    def point(self, residual=.02, visible=True, name='nose_tip'):
        return ProjectionPoint('front', name, residual, visible, True)

    def test_lost_visibility_is_not_an_improvement(self):
        self.assertEqual(screen((self.point(),), (self.point(.001, False),)).status, 'REGRESSION_FLAGGED')

    def test_residual_increase_flagged_and_improvement_not_certified(self):
        self.assertEqual(screen((self.point(),), (self.point(.05),)).status, 'REGRESSION_FLAGGED')
        self.assertEqual(screen((self.point(),), (self.point(.01),)).status, 'NO_REGRESSION_FLAGGED')

    def test_unsupported_jaw_is_not_acceptance_evidence(self):
        self.assertEqual(screen((self.point(name='jaw_0'),), ()).status, 'INSUFFICIENT_EVIDENCE')

    def test_nonfinite_and_duplicate_rejected(self):
        for points in [(self.point(float('nan')),), (self.point(), self.point())]:
            with self.assertRaises(ValueError):
                screen(points, ())
