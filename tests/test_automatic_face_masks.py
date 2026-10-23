import unittest
import numpy as np
from PIL import Image
from memento3d_head.adapters.automatic_face_masks import preprocess,decode_logits,restore_crop,crop_box,KEEP


class AutomaticMaskTests(unittest.TestCase):
    def test_rgb_normalization(self):
        tensor=preprocess(Image.new('RGB',(10,20),(255,0,128)))
        self.assertEqual(tensor.shape,(1,3,512,512))
        self.assertEqual(tensor.dtype,np.float32)
        np.testing.assert_allclose(tensor[0,:,0,0],[(1-.485)/.229,(0-.456)/.224,(128/255-.406)/.225],rtol=1e-6)

    def test_decode_and_class_policy(self):
        logits=np.zeros((1,19,512,512),np.float32);logits[:,17]=8
        labels,confidence,group=decode_logits(logits)
        self.assertTrue(np.all(labels==17));self.assertTrue(np.all(confidence>.99))
        self.assertNotIn(17,KEEP);self.assertNotIn(6,KEEP);self.assertNotIn(16,KEEP)
        self.assertIn(4,KEEP);self.assertIn(12,KEEP)
        logits[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):decode_logits(logits)

    def test_retained_class_boundary_does_not_create_hole(self):
        logits=np.full((1,19,512,512),-15,np.float32)
        logits[:,1]=0;logits[:,10]=0
        labels,confidence,group=decode_logits(logits)
        self.assertTrue(np.all(confidence<.55))
        self.assertTrue(np.all(group>.99))
        self.assertTrue(np.isin(labels,KEEP).all())

    def test_inverse_crop_padding(self):
        labels=np.full((8,8),17,np.uint8)
        restored=restore_crop(labels,(-4,-4,4,4),(10,10))
        self.assertTrue(np.all(restored[:4,:4]==17));self.assertTrue(np.all(restored[4:]==0))

    def test_square_crop_and_invalid_coordinates(self):
        coordinates=np.column_stack((np.linspace(.25,.75,200),np.linspace(.1,.9,200),np.ones(200)))
        x,y,x1,y1=crop_box(coordinates,(1000,800))
        self.assertEqual(x1-x,y1-y)
        with self.assertRaises(ValueError):crop_box(np.zeros((200,3)),(1000,800))


if __name__=='__main__':unittest.main()
