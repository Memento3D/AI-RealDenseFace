import tempfile,unittest
from pathlib import Path
import numpy as np,trimesh
from PIL import Image
from tools.customization.review_glb import header,replace_head,crop_proxy
from tools.customization.fit_trial import world_parts
from tools.headapi.assembly_math import AXES


def make_head(radius):
    mesh=trimesh.creation.icosphere(subdivisions=1,radius=radius);mesh.apply_translation([0,0,.9])
    uv=(mesh.vertices[:,:2]/radius+1)/2
    image=Image.fromarray(np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,0]]],np.uint8))
    mesh.visual=trimesh.visual.TextureVisuals(uv=uv,material=trimesh.visual.material.PBRMaterial(baseColorTexture=image,metallicFactor=0,roughnessFactor=1))
    return mesh


def save(scene,path):
    scene.apply_transform(np.linalg.inv(AXES));scene.export(path)


class ReviewGlbTests(unittest.TestCase):
    def test_stream_replacement_preserves_body_and_original_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base.glb';sidecar=root/'head.glb';target=root/'new.glb'
            body=trimesh.creation.box(extents=[.2,.1,.8]);body.apply_translation([0,0,.4]);body.visual.vertex_colors=[200,120,90,255]
            source=trimesh.Scene();source.add_geometry(body,node_name='Body',geom_name='Body');source.add_geometry(make_head(.05),node_name='Personalized_Complete_Head',geom_name='Personalized_Complete_Head')
            source.add_geometry(trimesh.creation.box(),node_name='Neck_Transition',geom_name='Neck_Transition');save(source,base)
            new=trimesh.Scene();new.add_geometry(make_head(.06),node_name='Personalized_Complete_Head',geom_name='Personalized_Complete_Head');save(new,sidecar)
            report=replace_head(base,sidecar,target)
            _,offset,size=header(base);_,new_offset,_=header(target)
            with base.open('rb') as a,target.open('rb') as b:
                a.seek(offset);b.seek(new_offset);self.assertEqual(a.read(size),b.read(size))
            before=world_parts(base);after=world_parts(target);expected=world_parts(sidecar)
            np.testing.assert_array_equal(before['Body'].vertices,after['Body'].vertices)
            np.testing.assert_array_equal(after['Personalized_Complete_Head'].vertices,expected['Personalized_Complete_Head'].vertices)
            self.assertNotIn('Neck_Transition',after);self.assertIn('Neck_Transition',report['removed_obsolete_connector_nodes'])

    def test_proxy_preserves_head_uv_texture_and_vertex_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'base.glb';proxy=root/'proxy.glb'
            body=trimesh.creation.box(extents=[.2,.1,.8]);body.apply_translation([0,0,.4]);body.visual.vertex_colors=[200,120,90,255]
            scene=trimesh.Scene();scene.add_geometry(body,node_name='Body',geom_name='Body');scene.add_geometry(make_head(.05),node_name='Personalized_Complete_Head',geom_name='Personalized_Complete_Head');save(scene,source)
            report=crop_proxy(source,proxy);before=world_parts(source);after=world_parts(proxy)
            a=before['Personalized_Complete_Head'];b=after['Personalized_Complete_Head']
            np.testing.assert_allclose(a.vertices[a.faces],b.vertices[b.faces],atol=1e-7)
            np.testing.assert_allclose(a.visual.uv[a.faces],b.visual.uv[b.faces],atol=1e-7)
            np.testing.assert_array_equal(np.asarray(a.visual.material.baseColorTexture),np.asarray(b.visual.material.baseColorTexture))
            np.testing.assert_array_equal(np.unique(after['Body'].visual.vertex_colors,axis=0),[[200,120,90,255]])
            self.assertLess(sum(c['proxy_triangles'] for c in report['components']),sum(c['source_triangles'] for c in report['components']))

if __name__=='__main__':unittest.main()
