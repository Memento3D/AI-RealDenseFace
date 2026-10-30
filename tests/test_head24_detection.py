import sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from head24_detection import scan_windows,map_detection,merge_detections,scan_faces,same_face
from head24_policy import choose_face,InputRejected

def landmarks(lo=(.2,.2),hi=(.6,.6)):
    return np.tile(np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]]),(119,1))

class DetectionTests(unittest.TestCase):
    def test_windows_cover_edges_and_respect_budget(self):
        windows=scan_windows((1920,2560));self.assertLessEqual(len(windows),64)
        for x,y in [(1,1),(1919,1),(1,2559),(1919,2559),(960,1280)]:
            self.assertTrue(any(a<=x<c and b<=y<d for a,b,c,d in windows))
    def test_tile_maps_to_native_not_detector_pixels(self):
        r=map_detection(landmarks(),(500,1000,1500,2000))
        self.assertTrue(np.allclose(r['box'],[700,1200,1100,1600]))
    def test_duplicate_exposures_and_windows_are_one_face(self):
        a=map_detection(landmarks(),(0,0,1000,1000));a['variant']='original'
        b=dict(a);b['points']=a['points']+1;b['box']=a['box']+1;b['variant']='shadow'
        clusters=merge_detections([a,b]);self.assertEqual(len(clusters),1);self.assertEqual(len(clusters[0]['observations']),2)
    def test_separate_people_are_not_merged(self):
        self.assertFalse(same_face([100,100,200,200],[230,100,330,200]))
    def test_cut_face_at_tile_edge_is_not_accepted(self):
        self.assertIsNone(map_detection(landmarks(lo=(0,.2)),(0,0,1000,1000)))
    def test_native_pixel_gate_is_not_lowered_by_mapping(self):
        r=map_detection(landmarks((.2,.2),(.25,.25)),(0,0,1000,1000))
        with self.assertRaisesRegex(InputRejected,'NO_FACE_MEETS_NATIVE_SIZE'):choose_face([r['box']],(1920,2560))
    def test_empty_detection_is_distinct_from_small_face(self):
        with self.assertRaisesRegex(InputRejected,'NO_FACE_DETECTED'):choose_face([],(1920,2560))
    def test_detection_only_brightening_does_not_change_source(self):
        image=Image.new('RGB',(256,320),(20,30,40));original=np.asarray(image).copy();means=[]
        def fake_detector(array):means.append(float(array.mean()));return []
        _,report=scan_faces(image,fake_detector)
        self.assertTrue(np.array_equal(np.asarray(image),original))
        self.assertGreater(max(means),min(means));self.assertFalse(report['source_colors_modified'])
        self.assertEqual(report['full_frame_original_faces'],0)

if __name__=='__main__':unittest.main()
