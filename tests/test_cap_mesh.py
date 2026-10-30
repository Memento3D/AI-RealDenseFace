import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('cap',Path(__file__).resolve().parents[1]/'scripts/add_graduation_cap.py')
cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap)


class CapTests(unittest.TestCase):
    def test_tube_is_closed_and_nondegenerate(self):
        triangles=cap.tube([[0.,0,0],[0,1.,0],[1,2.,0]],.1)
        v,ids=np.unique(triangles.reshape(-1,3),axis=0,return_inverse=True)
        f=ids.reshape(-1,3);edges=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
        _,count=np.unique(edges,axis=0,return_counts=True)
        self.assertTrue((count==2).all())
        self.assertTrue((np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)>1e-9).all())
