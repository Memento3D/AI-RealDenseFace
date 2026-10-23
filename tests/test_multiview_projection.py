import unittest
import numpy as np
from memento3d_head.adapters.multiview_projection import select_sources

class MultiviewSelectionTests(unittest.TestCase):
    def test_eligibility_unknown_and_tie_order(self):
        vertices=np.array([[0.,0.,1.],[0.,1.,1.],[1.,0.,1.]])
        faces=np.array([[0,1,2]]*4)
        view={'rotation_vector':[0.,0.,0.],'translation_mm':[0.,0.,10.]}
        views={s:view for s in ('front','left','right')}
        covered=np.array([[True,False,False,False],[True,True,False,False],[True,True,True,False]])
        result=select_sources(vertices,faces,views,covered)
        np.testing.assert_array_equal(result,[0,1,2,-1])

    def test_cli_dispatch(self):
        from memento3d_head.research_commands import COMMANDS
        self.assertIn('preview-multiview-color',COMMANDS)

    def test_build_preserves_positions_and_hides_unknown(self):
        import tempfile,json
        from pathlib import Path
        from PIL import Image
        from test_photo_projection import PhotoProjectionTests
        from memento3d_head.adapters.multiview_projection import build_preview
        v,f,view,landmarks=PhotoProjectionTests().fixture();before=v.copy()
        r={'views':{s:view for s in ('front','left','right')},'assessment':{'photos':[{'slot':s,'oriented_width':100,'oriented_height':100,'face':{'landmarks':landmarks}} for s in ('front','left','right')]}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);photos={}
            for slot in ('front','left','right'):
                photos[slot]=root/(slot+'.png');Image.new('RGB',(100,100),'red').save(photos[slot])
            info=build_preview(root,v,f,r,photos,{})
            self.assertEqual(info['covered_triangles'],1)
            with np.load(root/'multiview_projection.npz') as saved:
                np.testing.assert_array_equal(saved['vertices'],before)
                np.testing.assert_array_equal(saved['chosen_source'],[0,-1])
            html=(root/'multiview_preview.html').read_text()
            pages=json.loads(html.split('<script id="pages" type="application/json">')[1].split('</script>')[0])
            self.assertEqual(set(pages),{'front','left','right','combined'})
            for page in pages.values():
                payload=json.loads(page.split('<script id="model-data" type="application/json">')[1].split('</script>')[0])
                np.testing.assert_array_equal(np.array(payload['candidate']['vertices']),before[f].reshape(-1,3))
                self.assertEqual(len(payload['photo_faces']),1)

    def test_protected_missing_front_cannot_use_side(self):
        vertices=np.array([[0.,0.,1.],[0.,1.,1.],[1.,0.,1.]])
        faces=np.array([[0,1,2]]*3)
        view={'rotation_vector':[0.,0.,0.],'translation_mm':[0.,0.,10.]}
        result=select_sources(vertices,faces,{s:view for s in ('front','left','right')},
            np.array([[True,False,False],[True,True,True],[True,True,True]]),np.array([True,True,False]))
        np.testing.assert_array_equal(result,[0,-1,1])

    def test_side_color_offset_is_bounded_and_front_unchanged(self):
        from PIL import Image
        from memento3d_head.adapters.multiview_projection import balance_side_colors
        images=[Image.new('RGB',(10,10),color) for color in ((120,130,140),(30,40,50),(125,135,145))]
        coords=[np.tile([.5,.5,1.],(40,1)) for _ in range(3)]
        result,info=balance_side_colors(images,coords,np.ones((3,40),bool),np.zeros(40,bool))
        self.assertIs(result[0],images[0])
        self.assertEqual(info['left']['rgb_offset'],[15.,15.,15.])
        self.assertEqual(result[2].getpixel((5,5)),(120,130,140))
        _,info=balance_side_colors(images,coords,np.ones((3,40),bool),np.ones(40,bool))
        self.assertFalse(info['left']['applied'])

    def test_front_wins_even_when_side_faces_camera_better(self):
        vertices=np.array([[0.,0.,1.],[0.,1.,1.],[1.,0.,1.]])
        faces=np.array([[0,1,2]])
        front={'rotation_vector':[0.,1.,0.],'translation_mm':[0.,0.,10.]}
        side={'rotation_vector':[0.,0.,0.],'translation_mm':[0.,0.,10.]}
        result=select_sources(vertices,faces,{'front':front,'left':side,'right':side},np.ones((3,1),bool))
        np.testing.assert_array_equal(result,[0])

    def test_feather_never_adds_coverage_and_preserves_interior(self):
        from memento3d_head.adapters.multiview_projection import boundary_alpha
        faces=np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
        alpha=boundary_alpha(7,faces,np.array([False,True,True,True,True]))
        self.assertEqual(alpha[0],0.)
        self.assertEqual(alpha[1],0.)
        self.assertEqual(alpha[2],0.)
        self.assertTrue(np.all((alpha>=0)&(alpha<=1)))
        self.assertEqual(alpha[6],1.)
        np.testing.assert_array_equal(boundary_alpha(7,faces,np.ones(5,bool)),np.ones(7))
