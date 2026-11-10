"""Policy/routing tests. Inference and visual quality are not simulated as evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.headapi.assembly_config import base_only, validate_assets
from tools.headapi.service import sha


class BasePolicyTests(unittest.TestCase):
    def test_default_mode_keeps_old_assembly(self):
        self.assertFalse(base_only({'assembly':{}}))
        self.assertTrue(base_only({'assembly':{'output_mode':'customization_base'}}))

    def config(self, root):
        path=root/'asset';path.write_bytes(b'pinned synthetic dependency')
        return {'root':str(root),'face_render_policy':'strict_uv_v1',
            'presentation_policy':'harmonized_head_v1','paths':{'blender':str(path)},
            'assembly':{'policy':'graduation_modular_preview_v1','output_mode':'customization_base',
                'web_body_ratio':.25,'neck_cut_z':.765,
                'retained_nodes':['Body_Cuff_Prepared','Wrist_Lining_Join','Complete_Hand_Posed_Static'],
                'assets':{name:{'path':'asset','sha256':sha(path)} for name in
                    ['template','hair_library','reference_bundle','reference_pose','embedding','semantic_masks','uv_assets']}}}

    def test_base_policy_rejects_baked_cap_and_changed_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.config(root)
            self.assertTrue(validate_assets(config))
            bad=copy.deepcopy(config);bad['assembly']['retained_nodes'].append('Cap_Board')
            with self.assertRaisesRegex(ValueError,'component'):validate_assets(bad)
            (root/'asset').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'pin'):validate_assets(config)

    def test_unknown_mode_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.config(Path(tmp));config['assembly']['output_mode']='typo'
            with self.assertRaisesRegex(ValueError,'output mode'):validate_assets(config)


class BaseRoutingTests(unittest.TestCase):
    def case(self):
        import test_headapi_uv_demo as fixtures
        case=fixtures.UVDemoAPITests('test_scene_failure_is_not_done');case.setUp()
        case.service.config.update(face_render_policy='strict_uv_v1',
            assembly={'policy':'graduation_modular_preview_v1','output_mode':'customization_base'})
        return case

    def test_base_skips_hair_and_keeps_job_specific_scene_contract(self):
        case=self.case()
        try:
            with patch('tools.headapi.assembly_config.validate_assets',return_value=True):
                job=case.post().json()['job_id']
            self.assertTrue(all(not o['used_for_hair_cap'] for o in case.service.get(job)['observations']))
            calls=case.stages(job);case.service.run(job)
            row=case.service.get(job)
            self.assertNotIn('03_hair_cap',calls)
            self.assertIn('03b_uv_texture',calls);self.assertIn('04_scene_export',calls)
            self.assertFalse((case.service.jobs/job/'hair').exists())
            self.assertEqual(row['status'],'MVP_EXPORTED')
            self.assertFalse(row['hair_cap_exported']);self.assertFalse(row['customization_ready'])
            self.assertEqual(row['output_kind'],'customization_base')
        finally:case.tearDown()

    def test_base_scene_failure_does_not_publish_head_only_as_success(self):
        case=self.case()
        try:
            with patch('tools.headapi.assembly_config.validate_assets',return_value=True):
                job=case.post().json()['job_id']
            case.stages(job,'04_scene_export');case.service.run(job)
            row=case.service.get(job)
            self.assertNotEqual(row['status'],'MVP_EXPORTED')
            self.assertNotIn('scene_artifact',row)
        finally:case.tearDown()


class LibraryPathTests(unittest.TestCase):
    def test_traversal_and_external_file_rejected(self):
        from tools.customization.audit_library import contained
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Unsafe'):contained(tmp,'../private.png')
            with self.assertRaisesRegex(ValueError,'Missing'):contained(tmp,'missing.glb')


if __name__=='__main__':unittest.main()
