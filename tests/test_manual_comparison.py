import unittest
from unittest.mock import patch
import numpy as np
from memento3d_head.adapters.manual_contour_comparison import compare_arc
from memento3d_head.domain.contracts import ValidationError

class ManualComparisonTests(unittest.TestCase):
    def test_order_direction_and_alternate_long_arc(self):
        outline=[[0,0],[0,20],[20,20],[20,0]]
        manual=[[0,2],[0,10],[0,18]]
        forward=compare_arc(manual,[outline]);reverse=compare_arc(manual[::-1],[outline])
        self.assertLess(forward['mean_ordered_distance_px'],.5)
        self.assertAlmostEqual(forward['mean_ordered_distance_px'],reverse['mean_ordered_distance_px'],places=8)
        self.assertFalse(forward['correspondence_verified'])
        self.assertEqual(forward['candidate_arc_count'],2)

    def test_shifted_silhouette_has_measured_discrepancy(self):
        reference=[[0,2],[0,10],[0,18]]
        result=compare_arc(reference,[[[3,0],[3,20],[20,20],[20,0]]])
        self.assertAlmostEqual(result['mean_ordered_distance_px'],3.,delta=.1)

    def test_missing_or_collapsed_arc_fails(self):
        for outlines in ([],[[[0,0],[0,0],[0,0]]]):
            with self.assertRaises(ValidationError):compare_arc([[0,0],[1,1],[2,2]],outlines)

    def test_cli_dispatch(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.manual_comparison_commands.execute',return_value=0) as execute:
            self.assertEqual(main(['compare-manual-contours','--workspace','w','--validation-run','v',
                                   '--registration-run','r','--output','o']),0)
        execute.assert_called_once()
