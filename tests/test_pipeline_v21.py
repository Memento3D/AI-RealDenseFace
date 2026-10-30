"""Regression checks for identity, reliable bright skin and cavity contact."""
import json,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from figurine21_appearance import bright_reference,normalize_photo
from figurine21_fit import compact_fit
from figurine21_render import render_fast
from run_pipeline_v2 import glb
from add_graduation_cap import load,save
ROOT=Path(__file__).resolve().parents[1]

class PipelineV21Tests(unittest.TestCase):
    def settings(self):return json.loads((ROOT/'configs/pipeline-v21-04.json').read_text())
    def sample(self):
        rgb=np.full((100,100,3),255,np.uint8);labels=np.zeros((100,100),np.uint8)
        rgb[10:90,10:90]=[125,90,80];labels[10:90,10:90]=1
        rgb[25:70,30:65]=[195,162,154]
        rgb[30:36,40:48]=[255,255,255]
        return Image.fromarray(rgb),labels
    def test_bright_reference_ignores_background_shadow_and_clipped_highlights(self):
        im,labels=self.sample();color,mask,report=bright_reference(im,labels,self.settings()['bright_skin'])
        np.testing.assert_allclose(color*255,[195,162,154],atol=1)
        self.assertFalse(mask[0,0]);self.assertFalse(mask[32,43]);self.assertFalse(report['palette_quantized'])
    def test_no_skin_pixels_fails_instead_of_guessing(self):
        im,labels=self.sample()
        with self.assertRaisesRegex(ValueError,'Insufficient'):bright_reference(im,labels*0,self.settings()['bright_skin'])
    def test_source_feature_pixels_and_background_are_preserved(self):
        im,labels=self.sample();labels[40:48,20:30]=4;labels[65:72,40:60]=12
        src=np.asarray(im).copy();src[labels==4]=[40,30,20];src[labels==12]=[140,80,80];im=Image.fromarray(src)
        out,report=normalize_photo(im,labels,np.array([195,162,154])/255,self.settings()['bright_skin'])
        out=np.asarray(out);protected=np.isin(labels,[0,4,12])
        np.testing.assert_array_equal(out[protected],src[protected]);self.assertTrue(report['protected_pixels_identical'])
        self.assertGreater(out[50,15,0],src[50,15,0]);self.assertLessEqual(report['max_L_lift_applied'],9.00001)
    def shell(self):
        t=np.linspace(0,np.pi/2,30)
        outer=np.column_stack([12*np.sin(t),12*np.cos(t)])
        inner=np.column_stack([10.8*np.sin(t[::-1]),10.8*np.cos(t[::-1])])
        mesh=trimesh.creation.revolve(np.vstack([outer,inner]))
        mesh.vertices=mesh.vertices[:,[0,2,1]];mesh.invert()
        if mesh.volume<0:mesh.invert()
        return mesh
    def test_compact_shell_attaches_without_moving_source_head(self):
        head=trimesh.creation.icosphere(subdivisions=2,radius=10);before=head.vertices.copy()
        fitted,report=compact_fit(self.shell(),head,0,.25,self.settings()['compact_fit'])
        np.testing.assert_array_equal(head.vertices,before)
        self.assertTrue(fitted.is_volume);self.assertTrue(report['sampled_contact_proxy_pass'])
        self.assertLessEqual(report['sampled_inner_gap_p95_mm'],.08)
        self.assertLess(report['crown_added_height_mm'],.95)
    def test_missing_cavity_observations_are_not_replaced_by_an_invented_shell(self):
        from figurine2_hair import nearest_ray
        head=trimesh.creation.icosphere(subdivisions=2,radius=10)
        def missing(mesh,origins,directions):
            return nearest_ray(mesh,origins,directions) if mesh is head else np.full(len(origins),np.nan)
        with patch('figurine21_fit.nearest_ray',side_effect=missing):
            with self.assertRaisesRegex(ValueError,'Too many unresolved'):
                compact_fit(self.shell(),head,0,.25,self.settings()['compact_fit'])
    def test_nonfinite_and_excessive_compact_ratio_rejected(self):
        head=trimesh.creation.icosphere()
        for value in [float('nan'),float('inf'),.01,2]:
            with self.assertRaises(ValueError):compact_fit(self.shell(),head,0,value,self.settings()['compact_fit'])
        bad=self.settings()['compact_fit'];bad['inset_mm']=float('nan')
        with self.assertRaisesRegex(ValueError,'dimensions'):compact_fit(self.shell(),head,0,.25,bad)
    def test_unlit_photo_material_is_not_lit_a_second_time(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'lit.glb';b=Path(d)/'unlit.glb'
            glb(a,[('sphere',trimesh.creation.icosphere(subdivisions=2,radius=5),[.6,.5,.4,1])])
            doc,blob=load(a)
            for mat in doc['materials']:mat.setdefault('extensions',{})['KHR_materials_unlit']={}
            doc['extensionsUsed']=['KHR_materials_unlit'];save(b,doc,blob)
            original=np.asarray(render_fast(a,size=100));unlit=np.asarray(render_fast(b,size=100))
            np.testing.assert_allclose(unlit[50,50],[153,127,102],atol=2)
            self.assertGreater(unlit[50,50,0],original[50,50,0])

if __name__=='__main__':unittest.main()
