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

    def test_face_indices_keep_components_separate_when_vertex_positions_overlap(self):
        def sphere(count, center, radii):
            index=np.arange(count,dtype=float)
            y=1-2*(index+.5)/count
            angle=np.pi*(3-np.sqrt(5))*index
            ring=np.sqrt(1-y*y)
            return np.column_stack((ring*np.cos(angle),y,ring*np.sin(angle)))*radii+center

        counts=(3931,546,546)
        head=sphere(counts[0],np.zeros(3),np.ones(3))
        left_eye=sphere(counts[1],np.array([-.32,0,0]),np.array([.11,.09,.08]))
        right_eye=sphere(counts[2],np.array([.32,0,0]),np.array([.11,.09,.08]))
        head[0]=left_eye[0]
        vertices=np.vstack((
            head,
            left_eye,
            right_eye,
        ))
        vertices[counts[0]]=vertices[0]
        faces=[];offset=0
        for count in counts:
            ids=np.arange(offset+1,offset+count-1)
            faces.append(np.column_stack((np.full(len(ids),offset),ids,ids+1)))
            offset+=count
        faces=np.vstack(faces)

        labels,info=eyes.identify(vertices,faces)

        self.assertEqual(info['component_vertices'],list(counts))
        self.assertEqual(np.unique(labels).tolist(),[0,1,2])
        self.assertLess(info['eye_centers_native'][0][0],info['eye_centers_native'][1][0])
        with self.assertRaisesRegex(ValueError,'component sizes'):
            eyes.identify(vertices[faces].reshape(-1,3))
