"""Regressions for continuous eye UV, bounded local detail and exterior protection."""
import sys,json,tempfile,unittest,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from figurine23_appearance import prepared_front,continuous_eye_uv
from figurine23_hair import protected_shell_fit
from figurine_preview import accessor
ROOT=Path(__file__).resolve().parents[1]

class PipelineV23Tests(unittest.TestCase):
    def profile(self):return json.loads((ROOT/'configs/pipeline-v23-04.json').read_text())
    def eye_fixture(self):
        parts=[];labels=[]
        for component,x in [(1,-.5),(2,.5)]:
            mesh=trimesh.creation.icosphere(subdivisions=1,radius=.3);p=mesh.vertices[mesh.faces].reshape(-1,3)+[x,0,0]
            parts.append(p);labels.extend([component]*len(mesh.faces))
        p=np.vstack(parts);q=np.c_[.5+p[:,0]*.2,.5+p[:,1]*.2,np.ones(len(p))]
        face={'points':p,'components':np.array(labels),'images':[Image.fromarray(np.full((32,32,3),160,np.uint8))],
              'data':{'coordinates':np.tile(q,(3,1,1)).tolist()}}
        doc={'bufferViews':[],'accessors':[],'images':[],'textures':[],'nodes':[],'meshes':[],'materials':[]}
        for i,name in enumerate(['Eye_ModelXNegative','Eye_ModelXPositive']):
            doc['nodes'].append({'name':name,'mesh':i});doc['meshes'].append({'primitives':[{'attributes':{'TEXCOORD_0':0},'material':i}]})
            doc['materials'].append({'pbrMetallicRoughness':{}})
        return face,doc,bytearray()
    def test_shared_eye_vertices_have_identical_direct_uv(self):
        f,d,b=self.eye_fixture();before=f['points'].copy()
        with tempfile.TemporaryDirectory() as tmp:r=continuous_eye_uv(d,b,f,Path(tmp))
        self.assertTrue(all(x['welded_vertex_uv_max_discontinuity']==0 for x in r))
        self.assertEqual(len(d['textures']),2);np.testing.assert_array_equal(f['points'],before)
        for i in [0,1]:
            uv=accessor(d,b,d['meshes'][i]['primitives'][0]['attributes']['TEXCOORD_0']);self.assertTrue(np.isfinite(uv).all())
    def test_inconsistent_source_projection_at_shared_vertex_is_rejected(self):
        f,d,b=self.eye_fixture();f['data']['coordinates'][0][0][0]+=.03
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'discontinuous'):continuous_eye_uv(d,b,f,Path(tmp))
    def test_photo_eye_crop_out_of_frame_is_not_invented(self):
        f,d,b=self.eye_fixture();q=np.asarray(f['data']['coordinates']);q[0,:,0]+=.7;f['data']['coordinates']=q.tolist()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'leaves front'):continuous_eye_uv(d,b,f,Path(tmp))
    def test_local_detail_is_bounded_and_does_not_touch_remote_skin_or_neck(self):
        source=np.full((64,80,3),[120,95,85],np.uint8);labels=np.ones((64,80),np.uint8)
        labels[20:27,18:29]=4;labels[20:27,50:61]=5;source[18:20,18:29]=30;labels[45:,:]=14
        ref=np.array([195,162,154])/255;s=self.profile()['appearance23']['eyelid'];im,active,r=prepared_front(Image.fromarray(source),labels,ref,s)
        self.assertGreater(r['detail_pixels'],0);self.assertGreaterEqual(r['min_added_L'],-s['dark_L_limit']*s['strength']-1e-9)
        self.assertLessEqual(r['max_added_L'],s['bright_L_limit']*s['strength']+1e-9)
        np.testing.assert_array_equal(im[45:,:,:],np.tile([195,162,154],(19,80,1)))
        self.assertEqual(r['outside_feature_band_base_max_error_rgb8'],0)
    def test_constant_input_does_not_create_eyelid_detail(self):
        labels=np.ones((40,60),np.uint8);labels[12:18,12:22]=4;labels[12:18,38:48]=5
        source=Image.fromarray(np.full((40,60,3),150,np.uint8))
        _,_,r=prepared_front(source,labels,np.array([.7,.6,.5]),self.profile()['appearance23']['eyelid'])
        self.assertAlmostEqual(r['min_added_L'],0,places=10);self.assertAlmostEqual(r['max_added_L'],0,places=10)
    def test_no_visible_eye_semantics_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            prepared_front(Image.new('RGB',(30,30)),np.ones((30,30),np.uint8),np.array([.7,.6,.5]),self.profile()['appearance23']['eyelid'])
    def test_outer_radial_retraction_budget_and_input_geometry_preserved(self):
        hair=trimesh.creation.icosphere(subdivisions=2,radius=12);head=trimesh.creation.icosphere(subdivisions=2,radius=10)
        original=hair.vertices.copy();compact=hair.copy();compact.vertices*=.87
        cache={'inner':np.full(len(hair.vertices),9.)};s=self.profile()['compact_fit']
        with patch('figurine23_hair.shell_fit',return_value=(compact,{})):
            fitted,r=protected_shell_fit(hair,head,0,s,cache)
        self.assertLessEqual(r['measured_outer_retraction_max_mm'],s['max_outer_radial_shift_mm']+1e-8)
        self.assertGreater(fitted.volume,compact.volume);np.testing.assert_array_equal(hair.vertices,original)
    def test_invalid_exterior_budget_rejected(self):
        for x in [float('nan'),0,3]:
            s=self.profile()['compact_fit'];s['max_outer_radial_shift_mm']=x
            with self.assertRaises(ValueError):protected_shell_fit(None,None,0,s)

if __name__=='__main__':unittest.main()
