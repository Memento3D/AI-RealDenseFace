"""Synthetic geometric and publication regression tests; no inference claims."""
import unittest
import numpy as np
from tools.headapi.assembly_math import AXES, eye_alignment, fit_library_hair, transform_points
from tools.headapi.assembly_audit import compare_triangles


class AssemblyMathTests(unittest.TestCase):
    def test_eye_anchor_uses_new_head_and_uniform_scale(self):
        reference = np.array([[-1.,0,0],[-.5,0,0],[.5,0,0],[1,0,0], [0,2,1]])
        faces = np.array([[0,1,4],[2,3,4]])
        triangles = np.zeros(68, dtype=int)
        weights = np.tile([1.,0,0], (68,1))
        triangles[[42,45]] = 1
        weights[39] = weights[45] = [0,1,0]
        embedding = {'full_faces': triangles, 'full_weights': weights}
        current = reference * 1.2 + [2,3,4]
        pose = np.eye(4); pose[:3,:3] *= .7; pose[:3,3] = [0,0,1]
        matrix = eye_alignment(current, faces, embedding, reference, pose)
        np.testing.assert_allclose(transform_points(current, matrix),
                                   transform_points(reference, pose@AXES), atol=1e-12)
        gram = matrix[:3,:3].T @ matrix[:3,:3]
        np.testing.assert_allclose(gram, np.eye(3)*gram[0,0], atol=1e-12)

    def test_degenerate_eyes_are_rejected(self):
        v = np.zeros((5,3)); f = np.array([[0,1,2]])
        e = {'full_faces': np.zeros(68,dtype=int), 'full_weights': np.tile([1.,0,0],(68,1))}
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            eye_alignment(v,f,e,v,np.eye(4))

    def test_hair_fit_leaves_head_arrays_untouched(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(100,3)); target = source*1.1+[2,3,4]
        before = target.copy(); hair = source+1
        fitted, scale = fit_library_hair(hair,source,target)
        np.testing.assert_array_equal(target,before)
        np.testing.assert_allclose(scale,1.1)
        np.testing.assert_allclose(fitted,hair*1.1+[2,3,4])

    def test_extreme_hair_fit_is_rejected(self):
        source=np.random.default_rng(3).normal(size=(80,3))
        with self.assertRaisesRegex(ValueError,'outside'):
            fit_library_hair(source,source,source*3)

    def test_protected_triangle_reindexing_passes_but_distortion_fails(self):
        a=np.random.default_rng(8).normal(size=(20,3,3))
        b=a[::-1,::-1].copy()
        self.assertEqual(compare_triangles(a,b),0)
        b[0,0,0]+=.01
        with self.assertRaisesRegex(ValueError,'changed'):
            compare_triangles(a,b)

    def test_distinct_triangles_with_identical_centroids_match_by_corners(self):
        triangle=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        center=triangle.mean(0)
        a=np.array([triangle,(triangle-center)*1.1+center])
        self.assertEqual(compare_triangles(a,a[::-1]),0)


class AssemblyContractTests(unittest.TestCase):
    def test_unknown_scene_policy_fails_admission(self):
        import test_headapi as fixtures
        case=fixtures.APITests('test_mvp_is_default_upload_mode')
        case.setUp()
        try:
            case.service.config['assembly']={'policy':'unknown'}
            self.assertEqual(case.post().status_code,503)
            self.assertEqual(case.executor.pending,[])
        finally:case.tearDown()

    def test_opt_in_scene_command_and_failure_never_publish_old_head(self):
        import test_headapi_uv_demo as fixtures
        case=fixtures.UVDemoAPITests('test_scene_failure_is_not_done')
        case.setUp()
        try:
            case.service.config['face_render_policy']='strict_uv_v1'
            job=case.post().json()['job_id']
            # Validation is separately tested; this tests post-admission routing.
            case.service.config['assembly']={'policy':'graduation_modular_preview_v1'}
            calls=case.stages(job,'04_scene_export'); stage=case.service.stage;commands=[]
            def record(job_id,name,args):
                commands.append((name,[str(x) for x in args]));return stage(job_id,name,args)
            case.service.stage=record;case.service.run(job)
            command=next(a for n,a in commands if n=='04_scene_export')
            self.assertIn('tools.headapi.assembly_worker',command)
            self.assertNotEqual(case.service.get(job)['status'],'MVP_EXPORTED')
            self.assertNotIn('scene_artifact',case.service.get(job))
        finally:case.tearDown()


if __name__=='__main__':unittest.main()
