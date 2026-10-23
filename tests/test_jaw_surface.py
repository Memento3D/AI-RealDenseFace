import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from memento3d_head.adapters.jaw_surface import visible_face_chains,compare_open_arc
from memento3d_head.adapters.manual_contour_comparison import render_comparison

class JawSurfaceTests(unittest.TestCase):
    def fixture(self):
        v=np.array([[10,0,0],[-10,0,0],[0,10,0],[0,-10,0],[0,0,10],[0,0,-10]],float)
        f=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
        view={'camera_matrix':[[100,0,50],[0,100,50],[0,0,1]],'rotation_vector':[.2,.3,.1],'translation_mm':[0,0,100]}
        return v,f,view

    def test_visible_surface_edges_and_exclusion(self):
        v,f,view=self.fixture()
        chains,info=visible_face_chains(v,f,np.arange(6),np.array([],int),view)
        self.assertGreater(info['visible_edges'],0);self.assertTrue(chains)
        self.assertFalse(info['mask_cut_edges_used'])
        empty,stats=visible_face_chains(v,f,np.arange(6),np.arange(6),view)
        self.assertEqual(empty,[]);self.assertEqual(stats['candidate_edges'],0)

    def test_crop_does_not_invent_silhouette(self):
        v=np.array([[-10,-10,100],[10,-10,100],[10,10,100],[-10,10,100]],float)
        f=np.array([[0,1,2],[0,2,3]])
        view=self.fixture()[2];view['rotation_vector']=[0,0,0];view['translation_mm']=[0,0,0]
        chains,info=visible_face_chains(v,f,np.arange(4),np.array([],int),view)
        self.assertEqual(chains,[]);self.assertEqual(info['candidate_edges'],0)

    def test_full_mesh_occluder_blocks_face_edges(self):
        v,f,view=self.fixture()
        sheet=np.array([[-100,-100,-30],[100,-100,-30],[100,100,-30],[-100,100,-30]],float)
        vertices=np.vstack((v,sheet));faces=np.vstack((f,[[6,7,8],[6,8,9]]))
        chains,info=visible_face_chains(vertices,faces,np.arange(6),np.array([],int),view)
        self.assertGreater(info['candidate_edges'],0);self.assertEqual(info['visible_edges'],0)
        self.assertEqual(chains,[])

    def test_open_arc_never_wraps_or_bridges_components(self):
        manual=[[0,0],[0,5],[0,10]]
        result=compare_open_arc(manual,[[[2,0],[2,5],[2,10]]])
        self.assertAlmostEqual(result['mean_ordered_distance_px'],2.)
        reverse=compare_open_arc(manual[::-1],[[[2,0],[2,5],[2,10]]])
        self.assertAlmostEqual(result['mean_ordered_distance_px'],reverse['mean_ordered_distance_px'])
        self.assertEqual(compare_open_arc(manual,[])['status'],'NO_SUPPORTED_ARC')

    def test_unsupported_render_has_no_fabricated_mesh_line(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);photo=p/'photo.png';Image.new('RGB',(100,100),'white').save(photo)
            missing=compare_open_arc([[10,50],[20,60],[30,70]],[])
            render_comparison(photo,p/'out.png',[{'points_px':[[10,50],[20,60],[30,70]]}],
                              [{'baseline':missing,'candidate':missing}])
            with Image.open(p/'out.png') as image:self.assertEqual(image.size,(200,100))

    def test_cli_dispatch_requires_geometry_and_pinned_config(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.manual_comparison_commands.execute',return_value=0) as execute:
            argv=['compare-jaw-contours']
            for flag in ('workspace','validation-run','registration-run','head-run','shape-run','mica-config','output'):
                argv+=['--'+flag,flag]
            self.assertEqual(main(argv),0)
        execute.assert_called_once()

    def test_short_chin_or_mouth_stub_is_rejected_without_distance(self):
        manual=[[0,0],[0,50],[0,100]]
        for chain in ([[0,80],[0,90],[0,100]],[[30,40],[31,41],[32,42]]):
            result=compare_open_arc(manual,[chain])
            self.assertEqual(result['status'],'NO_SUPPORTED_ARC')
            self.assertNotIn('mean_ordered_distance_px',result)
            self.assertNotIn('mesh_samples_px',result)
            self.assertTrue(result['rejected_arcs'])

    def test_disconnected_partial_chains_are_not_bridged(self):
        result=compare_open_arc([[0,0],[0,50],[0,100]],
                                [[[1,0],[1,30],[1,40]],[[1,60],[1,70],[1,100]]])
        self.assertEqual(result['status'],'NO_SUPPORTED_ARC')

    def test_loop_with_good_endpoints_but_wrong_path_is_rejected(self):
        result=compare_open_arc([[0,0],[0,50],[0,100]],
                                [[[0,0],[80,0],[80,100],[0,100]]])
        self.assertEqual(result['status'],'NO_SUPPORTED_ARC')

    def test_matching_chain_wins_over_stub_without_mutating_inputs(self):
        reference=[[0,0],[0,50],[0,100]]
        chains=[[[0,80],[0,90],[0,100]],[[2,0],[2,50],[2,100]]]
        import copy
        before=copy.deepcopy(chains)
        result=compare_open_arc(reference,chains)
        self.assertEqual(result['status'],'UNVERIFIED_ARC_PROPOSED')
        self.assertEqual(result['supported_arc_count'],1)
        self.assertAlmostEqual(result['mean_ordered_distance_px'],2.)
        self.assertFalse(result['support']['policy_calibrated'])
        self.assertEqual(chains,before)
