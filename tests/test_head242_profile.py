import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from head24_policy import inference_box,bbox_gate,InputRejected
from detect_head24_profile import map_box

class ProfileTests(unittest.TestCase):
    def test_profile_explicit_roi_without_fake_landmarks(self):
        row={'landmarks_crop':None,'roi_backend':'facebox','roi_score':.9,'face_box_crop':[300,300,700,700]}
        self.assertTrue(np.array_equal(inference_box(row,'right'),[300,300,700,700]));self.assertIsNone(row['landmarks_crop'])
    def test_front_cannot_use_landmark_free_fallback(self):
        with self.assertRaises(InputRejected):inference_box({'roi_backend':'facebox','roi_score':.99,'face_box_crop':[300,300,700,700]},'front')
    def test_low_score_profile_rejected(self):
        with self.assertRaises(InputRejected):inference_box({'roi_backend':'facebox','roi_score':.59,'face_box_crop':[300,300,700,700]},'right')
    def test_profile_missing_or_reversed_roi_rejected(self):
        for box in (None,[700,300,300,700],[-1,0,500,500]):
            with self.assertRaises(InputRejected):inference_box({'roi_backend':'facebox','roi_score':.9,'face_box_crop':box},'left')
    def test_detector_padding_inverse_maps_native_coordinates(self):
        p=map_box([160,80,200,125],.9,[0,0,1920,2560],2560,False)
        self.assertTrue(np.allclose(p['box'],[800,400,1000,625]))
    def test_padding_detection_rejected(self):
        self.assertIsNone(map_box([400,80,500,125],.9,[0,0,1920,2560],2560,False))
    def test_score_threshold_preserved(self):
        self.assertIsNone(map_box([160,80,200,125],.59,[0,0,1920,2560],2560,False))
    def test_wrong_body_scale_fails_profile_gate(self):
        p=np.tile(np.array([[0,0],[900,0],[0,900],[900,900]]),(30,1))
        with self.assertRaises(InputRejected):bbox_gate(p,[300,300,400,400])
    def test_correct_profile_roi_passes_coarse_gate(self):
        p=np.tile(np.array([[300,300],[600,300],[300,650],[600,650]]),(30,1))
        self.assertLess(bbox_gate(p,[300,300,600,650])['center_error_in_roi_diagonals'],1e-10)

if __name__=='__main__':unittest.main()
