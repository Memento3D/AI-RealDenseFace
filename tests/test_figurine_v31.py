"""Requirements for continuous hair and photo-candidate eye appearance."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import numpy as np
    import trimesh
    import manifold3d
    from figurine_surface31 import ray_distances,cap_mesh,continuous_hair,photo_eye_direction,painted_eye
    from figurine_geometry import weld
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'optional figurine CPU dependencies unavailable')
class ContinuousHairTests(unittest.TestCase):
    def test_grooves_cannot_consume_the_base(self):
        import json
        from run_figurine_v3 import validate_config
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/figurine04.v31.json').read_text())
        c['hair']['groove_mm']=c['hair']['base_mm']
        with self.assertRaisesRegex(ValueError,'groove'):
            validate_config(c)

    def test_rays_intersect_known_box_surface(self):
        mesh=trimesh.creation.box(extents=[2,4,6])
        np.testing.assert_allclose(ray_distances(mesh,np.zeros(3),np.eye(3)),[1,2,3],atol=1e-8)

    def test_missed_rays_do_not_create_invented_scalp(self):
        with self.assertRaises(ValueError):
            ray_distances(trimesh.creation.box(),np.array([4.,4,4]),np.array([[1.,0,0]]))

    def test_missing_views_do_not_punch_cap(self):
        head=trimesh.creation.icosphere(subdivisions=2,radius=10)
        camera=np.array([[10,0,0,10000],[0,10,0,10000],[0,0,0,1.]])
        mask=np.zeros((16,16),bool);depth=np.full((16,16),np.inf)
        config=dict(base_mm=.7,ridge_mm=.25,anchor_mm=.7,sweep=.8,part=.15,clumps=8)
        hair,report=continuous_hair(head,0,[camera],[mask],[depth],config)
        self.assertTrue(hair.is_volume)
        self.assertEqual(len(hair.split()),1)
        self.assertEqual(report['mask_fitted_columns'],0)
        self.assertGreaterEqual(report['min_sampled_radial_thickness_mm'],.7)
        # A complete periodic boundary exists even with no photo support.
        boundary=np.array(report['hairline_phi_degrees'])
        self.assertEqual(len(boundary),128)
        self.assertLess(abs(boundary[0]-boundary[-1]),2)

    def test_eye_fallback_is_explicit_and_normalized(self):
        from PIL import Image
        camera=np.array([[100,0,0,256],[0,100,0,256],[0,0,1,10.]])
        direction,report=photo_eye_direction(Image.new('RGB',(512,512)),np.zeros((512,512),np.uint8),camera,np.zeros(3),np.ones(3))
        self.assertAlmostEqual(np.linalg.norm(direction),1)
        self.assertEqual(report['method'],'camera-facing fallback')

    def test_eye_texture_survives_glb_and_eye_can_be_welded_for_print(self):
        import io
        from PIL import Image
        from figurine_preview import add_mesh,accessor
        config=dict(gaze_yaw_deg=0,gaze_pitch_deg=0,iris_ratio=.65,pupil_ratio=.48,highlight_ratio=.055,sclera_rgb=[192,183,170],iris_rgb=[58,39,27])
        eye,colors=painted_eye(np.zeros(3),np.ones(3),config,np.array([0.,0,1]),0)
        self.assertTrue(weld(eye.vertices[eye.faces]).is_volume)
        png=Image.open(io.BytesIO(eye.metadata['eye_texture_png']))
        self.assertEqual(png.size,(1024,512))
        doc=dict(bufferViews=[],accessors=[],materials=[],meshes=[],nodes=[],scenes=[dict(nodes=[])])
        blob=bytearray();add_mesh(doc,blob,'Eye_Test',eye,colors)
        p=doc['meshes'][0]['primitives'][0]
        uv=accessor(doc,blob,p['attributes']['TEXCOORD_0'])
        self.assertEqual(len(uv),len(eye.vertices))
        self.assertTrue(np.isfinite(uv).all())
        self.assertIn('baseColorTexture',doc['materials'][0]['pbrMetallicRoughness'])

if __name__=='__main__':unittest.main()
