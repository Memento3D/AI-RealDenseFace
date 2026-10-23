import unittest
import numpy as np
from memento3d_head.adapters.anatomical_jaw import lower_face_boundary,project_anatomical_jaw,compare_anatomical_arc
from memento3d_head.domain.contracts import ValidationError

class AnatomicalJawTests(unittest.TestCase):
    def fixture(self):
        v=np.array([[x,y,10.] for y in (-1.,0.,1.) for x in (-1.,0.,1.)])
        f=[]
        for row in range(2):
            for col in range(2):
                a=row*3+col;f.extend([[a,a+1,a+3],[a+1,a+4,a+3]])
        return v,np.array(f),np.arange(9)

    def test_lower_patch_boundary_continuous_without_internal_edges(self):
        v,f,face=self.fixture();ids,info=lower_face_boundary(v,f,face)
        self.assertEqual(set(ids),{0,1,2,3,5})
        self.assertEqual(info['boundary_loops'],1)
        self.assertFalse(info['is_occluding_silhouette'])
        self.assertNotIn(4,ids)

    def test_frozen_boundary_survives_candidate_deformation_and_does_not_modify_mesh(self):
        v,f,face=self.fixture();boundary=lower_face_boundary(v,f,face)
        changed=v.copy();changed[3,1]=.01
        view={'camera_matrix':[[100.,0,50],[0,100.,50],[0,0,1]],'rotation_vector':[0.,0,0],
              'translation_mm':[0.,0,0]}
        before=changed.copy()
        chains,info=project_anatomical_jaw(changed,f,face,view,boundary=boundary)
        self.assertEqual(info['boundary_vertices'],boundary[0]);self.assertTrue(info['boundary_frozen_from_baseline'])
        self.assertEqual(len(chains),1);np.testing.assert_array_equal(before,changed)

    def test_unsupported_proxy_is_drawable_but_has_no_fitting_metric(self):
        result=compare_anatomical_arc([[0,0],[0,50],[0,100]],[[[15,10],[15,50],[15,90]]])
        self.assertEqual(result['status'],'ANATOMICAL_PROXY_INSUFFICIENT_SUPPORT')
        self.assertIn('diagnostic_mesh_samples_px',result)
        self.assertNotIn('mesh_samples_px',result);self.assertNotIn('mean_ordered_distance_px',result)
        self.assertFalse(result['projection_visibility_verified'])

    def test_supported_and_reversed_proxy(self):
        ref=[[0,0],[0,50],[0,100]];chains=[[[2,0],[2,50],[2,100]]]
        one=compare_anatomical_arc(ref,chains);two=compare_anatomical_arc(ref[::-1],chains)
        self.assertEqual(one['status'],'ANATOMICAL_PROXY_SUPPORTED')
        self.assertAlmostEqual(one['mean_ordered_distance_px'],two['mean_ordered_distance_px'])
        self.assertFalse(one['correspondence_verified'])

    def test_invalid_or_missing_patch_fails(self):
        v,f,face=self.fixture()
        with self.assertRaises(ValidationError):lower_face_boundary(v,f,np.array([99]))
        with self.assertRaises(ValidationError):lower_face_boundary(v,f,np.array([4]))
