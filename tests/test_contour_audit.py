import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from memento3d_head.adapters.contour_audit import select_contour,mesh_visibility,interpolate,draw_audit
from memento3d_head.domain.contracts import ValidationError

class ContourAuditTests(unittest.TestCase):
    def table(self):
        return np.tile(np.arange(17),(79,1)),np.tile([1.,0.,0.],(79,17,1))

    def test_camera_axis_conversion_and_signed_table_indices(self):
        import cv2
        faces,weights=self.table()
        for yaw,row,clamped in [(0,0,False),(20,20,False),(-20,59,False),(60,39,True),(-60,78,True)]:
            ry=cv2.Rodrigues(np.array([0.,np.radians(yaw),0.]))[0]
            camera=np.diag([1.,-1.,-1.])@ry
            vector=cv2.Rodrigues(camera)[0].reshape(3)
            _,_,selection=select_contour(vector,faces,weights)
            self.assertEqual(selection['table_row'],row)
            self.assertEqual(selection['angle_clamped'],clamped)
            self.assertAlmostEqual(selection['camera_yaw_degrees'],yaw,places=5)

    def test_invalid_table_rejected(self):
        with self.assertRaises(ValidationError):select_contour([0.,0.,0.],np.zeros((78,17)),np.zeros((79,17,3)))

    def test_ray_occlusion_distinguishes_front_and_hidden_surface(self):
        vertices=np.array([[-3.,-3.,5.],[3.,-3.,5.],[3.,3.,5.],[-3.,3.,5.]])
        faces=np.array([[0,1,2],[0,2,3]])
        points=np.array([[0.,0.,10.],[0.,0.,5.],[10.,0.,10.],[0.,0.,-1.]])
        before=points.copy()
        rows=mesh_visibility(vertices,faces,points,np.zeros(3),np.zeros(3))
        self.assertTrue(rows[0]['self_occluded']);self.assertFalse(rows[0]['mesh_visible'])
        self.assertTrue(rows[1]['mesh_visible']);self.assertTrue(rows[2]['mesh_visible'])
        self.assertFalse(rows[3]['positive_depth'])
        np.testing.assert_array_equal(points,before)

    def test_barycentric_points_stay_on_selected_triangle(self):
        vertices=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        point=interpolate(vertices,np.array([[0,1,2]]),[0],np.array([[.5,.25,.25]]))
        np.testing.assert_allclose(point,[[.5,.5,0.]])

    def test_diagnostic_image_handles_unsupported_angle_without_fit(self):
        rows=[{'observed_px':[40+i*2,50+i],'projected_px':[41+i*2,52+i],
               'self_occluded':i%2==0} for i in range(24)]
        audit={'selection':{'camera_yaw_degrees':50.,'table_yaw_degrees':39,'angle_clamped':True},'landmarks':rows}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);Image.new('RGB',(300,200),'white').save(root/'photo.png')
            draw_audit(root/'photo.png',root/'audit.png',audit,audit)
            with Image.open(root/'audit.png') as image:self.assertEqual(image.size,(600,285))

    def test_top_level_dispatch(self):
        from memento3d_head.bootstrap import main
        with patch('memento3d_head.contour_commands.execute',return_value=0) as execute:
            self.assertEqual(main(['audit-contours','--registration-run','run','--head-run','head',
                                   '--shape-run','shape','--mica-config','mica.json','--output','out']),0)
        execute.assert_called_once()
