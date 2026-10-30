import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('eye_components',Path(__file__).resolve().parents[1]/'scripts/eye_components.py')
eyes=importlib.util.module_from_spec(spec);spec.loader.exec_module(eyes)


class EyeComponentTests(unittest.TestCase):
    def test_unknown_topology_is_rejected_instead_of_splitting_by_position(self):
        vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[-1,0,0],[-2,0,0],[-1,1,0]])
        with self.assertRaisesRegex(ValueError,'component sizes'):
            eyes.identify(vertices)
