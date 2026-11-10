"""Decoded GLB audit for the personalized face, independent of Blender state."""
import argparse,io,json,struct
from pathlib import Path
import numpy as np
from PIL import Image
from .service import sha,write

def face(path):
    raw=Path(path).read_bytes()
    if len(raw)<28 or struct.unpack_from('<4sII',raw)!=(b'glTF',2,len(raw)):raise ValueError('Invalid GLB header')
    n,kind=struct.unpack_from('<II',raw,12)
    if kind!=0x4e4f534a:raise ValueError('Missing JSON chunk')
    doc=json.loads(raw[20:20+n]);m,kind=struct.unpack_from('<II',raw,20+n)
    if kind!=0x004e4942:raise ValueError('Missing binary chunk')
    binary=memoryview(raw)[28+n:28+n+m]
    if doc.get('extensionsRequired'):raise ValueError('Web requires an extension decoder')
    def access(index):
        a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
        if 'sparse' in a or a.get('normalized'):raise ValueError('Unsupported face accessor')
        dtype=np.dtype({5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']]);width={'VEC2':2,'VEC3':3,'SCALAR':1}[a['type']]
        return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',width*dtype.itemsize),dtype.itemsize))
    matched=[]
    def walk(index,parent):
        node=doc['nodes'][index]
        if 'matrix' in node:local=np.array(node['matrix']).reshape(4,4).T
        else:
            x,y,z,w=node.get('rotation',[0,0,0,1]);local=np.eye(4)
            local[:3,:3]=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])@np.diag(node.get('scale',[1,1,1]));local[:3,3]=node.get('translation',[0,0,0])
        world=parent@local
        if 'mesh' in node:
            for primitive in doc['meshes'][node['mesh']]['primitives']:
                material=doc['materials'][primitive['material']]
                if material.get('name','').startswith('Khoa_Face_Photo'):
                    if primitive.get('mode',4)!=4:raise ValueError('Face is not triangles')
                    indices=access(primitive['indices']).ravel();pos=access(primitive['attributes']['POSITION'])[indices].astype(float)
                    pos=pos@world[:3,:3].T+world[:3,3];uv=access(primitive['attributes']['TEXCOORD_0'])[indices]
                    corners=np.column_stack([pos,uv]).reshape(-1,15)
                    # Exporter preserves triangle winding/order for this pinned template.
                    image=doc['images'][doc['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]['source']]
                    view=doc['bufferViews'][image['bufferView']];start=view.get('byteOffset',0)
                    rgb=np.asarray(Image.open(io.BytesIO(binary[start:start+view['byteLength']])).convert('RGB'))
                    matched.append((corners,rgb))
        for child in node.get('children',[]):walk(child,world)
    for index in doc['scenes'][doc.get('scene',0)]['nodes']:walk(index,np.eye(4))
    if len(matched)!=1:raise ValueError('Expected exactly one personalized face primitive')
    return matched[0]

def audit(master,web):
    a,ia=face(master);b,ib=face(web)
    if a.shape!=b.shape or not np.allclose(a,b,atol=1e-7,rtol=0):raise ValueError('Personalized face triangles/world positions/UV changed')
    if not np.array_equal(ia,ib):raise ValueError('Personalized face atlas changed')
    return {'status':'PERSONALIZED_FACE_WEB_AUDIT_PASSED','master_sha256':sha(master),'web_sha256':sha(web),'face_triangles':len(a),'world_position_and_uv_tolerance':1e-7,'face_atlas_decoded_pixels_identical':True,'display_only':True,'ready_for_machine':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--master',required=True);p.add_argument('--web',required=True);p.add_argument('--report',required=True);a=p.parse_args();report=audit(a.master,a.web);write(a.report,report);print(report['status'])
