"""Projection, conservative color correction and review binding checks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from memento3d_head.adapters.head_texture_workspace import source_visibility,estimate_gains,sample,srgb_linear,build_workspace


class HeadWorkspaceTests(unittest.TestCase):
    def test_pixel_coordinate_round_trip(self):
        image=np.arange(4*4*3).reshape(4,4,3)/255
        coords=np.array([[.25, .5,1.]])
        np.testing.assert_allclose(sample(image,coords)[0],image[2,1])

    def test_mask_excludes_visible_triangle(self):
        cam=np.array([[-.2,-.2,1.],[.2,-.2,1.],[0,.2,1.]])
        faces=np.array([[0,2,1]])
        k=np.array([[100.,0,50],[0,100,50],[0,0,1]])
        xy=cam@k.T;xy=xy[:,:2]/xy[:,2,None]
        coords=np.column_stack((xy[:,0]/100*cam[:,2],(1-xy[:,1]/100)*cam[:,2],cam[:,2]))
        self.assertTrue(source_visibility(coords,faces,k,(100,100),np.ones((100,100),bool))[0][0])
        self.assertFalse(source_visibility(coords,faces,k,(100,100),np.zeros((100,100),bool))[0][0])

    def test_gains_bounded_and_low_overlap_left_unchanged(self):
        colors=np.full((3,60,3),.5);colors[1]=.2;colors[2]=.8
        eligible=np.ones((3,60),bool);eligible[2,5:]=False
        gains,counts=estimate_gains(colors,eligible)
        np.testing.assert_allclose(gains[0],1)
        np.testing.assert_allclose(gains[1],1.18)
        np.testing.assert_allclose(gains[2],1)
        self.assertEqual(counts,[60,5])
        self.assertTrue(np.all(srgb_linear(np.array([0.,.5,1.]))==np.array([0.,srgb_linear(np.array([.5]))[0],1.])))

    def test_workspace_and_stale_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);texture=root/'three_textures_test';texture.mkdir()
            state=root/'joint.flame.npz';np.savez(state,image_width=16,image_height=16,camera_fov_y=30.)
            parent=root/'report.json';parent.write_text('{}')
            from memento3d_head.adapters.head_texture_workspace import digest
            source_report=texture/'report.json';source_report.write_text(json.dumps({'status':'THREE_TEXTURES_PREVIEW_EXPORTED','input_sha256':{str(state):digest(state),str(parent):digest(parent)}}))
            image=Image.new('RGB',(16,16),(170,120,100));import base64,io
            stream=io.BytesIO();image.save(stream,format='PNG');url='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
            vertices=np.zeros((5023,3));vertices[:,0]=np.linspace(-1,1,5023);vertices[:,1]=np.linspace(-.5,.5,5023)**2;vertices[:,2]=.1
            faces=np.tile(np.array([[0,2,1]]),(9976,1))
            coords=np.tile(np.array([[.5,.5,1.]]),(5023,1))
            data={'baseline':{'vertices':vertices.tolist()},'faces':faces.tolist(),'layers':[{'image':url,'coordinates':coords.tolist(),'faces':faces.tolist()} for _ in range(3)]}
            viewer=texture/'head.three_textures.html';viewer.write_text('<script id="model-data" type="application/json">'+json.dumps(data)+'</script>')
            def visibility(*args):return np.ones(9976,bool),np.ones(5023)
            with patch('memento3d_head.adapters.head_texture_workspace.source_visibility',visibility):
                out,report=build_workspace(root,texture)
            self.assertTrue((out/'head.workspace.html').exists())
            self.assertTrue((out/'head.masks.html').exists())
            self.assertFalse(report['geometry_modified'])
            self.assertEqual(report['status'],'TEXTURE_REVIEW_READY')
            bad=root/'bad.json';bad.write_text(json.dumps({'schema_version':1,'source_viewer_sha256':'stale','masks':{}}))
            with self.assertRaisesRegex(ValueError,'another texture source'):build_workspace(root,texture,bad)


if __name__=='__main__':unittest.main()
