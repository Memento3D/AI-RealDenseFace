import base64
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('workspace_export', Path(__file__).resolve().parents[1]/'scripts/export_workspace_glb.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def test_masked_face_uses_skin_and_keeps_geometry(self):
        b=io.BytesIO();Image.new('RGB',(8,8),(255,0,0)).save(b,format='PNG')
        photo='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
        vertices=[[0,0,0],[1,0,0],[0,1,0]]
        data=dict(vertices=vertices,normals=[[0,0,1]]*3,weights=[[0,0,0]]*3,
            coordinates=[[[0,0,1],[1,0,1],[0,1,1]]]*3,images=[photo]*3,
            gains=[[1,1,1]]*3,skin=[.5,.4,.3])
        with tempfile.TemporaryDirectory() as t:
            path=Path(t);source=path/'head.workspace.html';out=path/'head.glb'
            source.write_text('<script id="data" type="application/json">'+json.dumps(data)+'</script>')
            exporter.export(source,out,16)
            raw=out.read_bytes()
            self.assertEqual(struct.unpack('<III',raw[:12]),(0x46546c67,2,len(raw)))
            size=struct.unpack('<I',raw[12:16])[0];doc=json.loads(raw[20:20+size]);start=20+size+8
            view=doc['bufferViews'][0]
            actual=np.frombuffer(raw[start+view['byteOffset']:start+view['byteOffset']+view['byteLength']],dtype='<f4').reshape(-1,3)
            np.testing.assert_array_equal(actual,vertices)
            view=doc['bufferViews'][-1]
            image=Image.open(io.BytesIO(raw[start+view['byteOffset']:start+view['byteOffset']+view['byteLength']]))
            self.assertEqual(image.getpixel((4,4)),(128,102,76))
            with self.assertRaises(ValueError): exporter.export(source,out,16)

    def test_rejects_wrong_viewer(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'old.html').write_text('<html></html>')
            with self.assertRaises(ValueError): exporter.export(p/'old.html',p/'bad.glb')
