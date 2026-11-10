import io,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from tools.headapi.production_audit import audit

def fixture(path,translation=0,color=(180,130,100)):
    positions=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='<f4').tobytes();uv=np.array([[0,0],[1,0],[0,1]],dtype='<f4').tobytes();indices=np.array([0,1,2],dtype='<u4').tobytes()
    stream=io.BytesIO();Image.new('RGB',(2,2),color).save(stream,format='PNG');png=stream.getvalue();binary=positions+uv+indices+png
    doc={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(binary)}],'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':36},{'buffer':0,'byteOffset':36,'byteLength':24},{'buffer':0,'byteOffset':60,'byteLength':12},{'buffer':0,'byteOffset':72,'byteLength':len(png)}],'accessors':[{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},{'bufferView':1,'componentType':5126,'count':3,'type':'VEC2'},{'bufferView':2,'componentType':5125,'count':3,'type':'SCALAR'}],'materials':[{'name':'Khoa_Face_Photo','pbrMetallicRoughness':{'baseColorTexture':{'index':0}}}],'textures':[{'source':0}],'images':[{'bufferView':3,'mimeType':'image/png'}],'meshes':[{'primitives':[{'attributes':{'POSITION':0,'TEXCOORD_0':1},'indices':2,'material':0}]}],'nodes':[{'mesh':0,'translation':[translation,0,0]}],'scenes':[{'nodes':[0]}],'scene':0}
    text=json.dumps(doc).encode();text+=b' '*((-len(text))%4);binary+=b'\0'*((-len(binary))%4)
    path.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(text)+len(binary))+struct.pack('<II',len(text),0x4e4f534a)+text+struct.pack('<II',len(binary),0x004e4942)+binary)

class WebFaceAuditTests(unittest.TestCase):
    def test_identical_decoded_face_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.glb';b=Path(tmp)/'b.glb';fixture(a);fixture(b)
            self.assertEqual(audit(a,b)['face_triangles'],1)
    def test_node_transform_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.glb';b=Path(tmp)/'b.glb';fixture(a);fixture(b,translation=.001)
            with self.assertRaisesRegex(ValueError,'world positions'):audit(a,b)
    def test_changed_face_pixels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.glb';b=Path(tmp)/'b.glb';fixture(a);fixture(b,color=(181,130,100))
            with self.assertRaisesRegex(ValueError,'atlas changed'):audit(a,b)

if __name__=='__main__':unittest.main()
