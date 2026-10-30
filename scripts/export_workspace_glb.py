"""Bake optimized head workspace textures into a standalone GLB (NumPy/Pillow)."""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct

import numpy as np
from PIL import Image


def linear(c):
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def sample(image, uv):
    h, w = image.shape[:2]
    x = np.clip(uv[:, 0] * w - .5, 0, w - 1)
    y = np.clip((1 - uv[:, 1]) * h - .5, 0, h - 1)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    jx, jy = np.minimum(ix + 1, w - 1), np.minimum(iy + 1, h - 1)
    a, b = (x - ix)[:, None], (y - iy)[:, None]
    return (image[iy, ix]*(1-a)*(1-b) + image[iy, jx]*a*(1-b)
            + image[jy, ix]*(1-a)*b + image[jy, jx]*a*b)


def export(source, output, cell=32, refine=False, separate_eyes=False, harmonize=False):
    if output.exists():
        raise ValueError('Output already exists; choose another --output path')
    raw = source.read_bytes()
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', raw.decode('utf-8-sig'), re.S)
    if not match:
        raise ValueError('Expected head.workspace.html, not the earlier texture viewer')
    d = json.loads(match.group(1))
    vertices = np.asarray(d['vertices'], dtype=np.float32)
    normals = np.asarray(d['normals'], dtype=np.float32)
    weights = np.asarray(d['weights'], dtype=float).reshape(-1, 3, 3)
    coords = np.asarray(d['coordinates'], dtype=float).reshape(3, -1, 3, 3)
    count = len(vertices) // 3
    if vertices.shape != (count*3, 3) or normals.shape != vertices.shape or weights.shape != (count,3,3) or coords.shape != (3,count,3,3):
        raise ValueError('Unexpected workspace shapes')
    if not all(np.isfinite(x).all() for x in (vertices, normals, weights, coords)):
        raise ValueError('Non-finite workspace data')
    images = [np.asarray(Image.open(io.BytesIO(base64.b64decode(s.split(',',1)[1]))).convert('RGB'), dtype=float)/255 for s in d['images']]
    gains = np.asarray(d['gains'], dtype=float)
    skin = np.asarray(d['skin'], dtype=float)
    corner_gains=None;source_bias=None;calibration_report=None;feature_report=None
    if harmonize:
        from eye_components import identify
        from color_harmonization import estimate
        component,_=identify(vertices)
        # Calibrate before feature source changes, using original visibility.
        corner_gains,source_bias,calibration_report=estimate(vertices,coords,weights,images,gains,skin,sample,linear,component)
        centers=vertices.reshape(-1,3,3).mean(1)
        height=np.ptp(vertices[:,1]);width=np.ptp(vertices[:,0]);eye_y=vertices[np.repeat(component>0,3),1].mean()
        center_x=(vertices[:,0].min()+vertices[:,0].max())/2
        x=(centers[:,0]-center_x)/width
        y=(centers[:,1]-eye_y)/height
        # Smooth source preference avoids a rectangular hard switch at cheeks.
        eye_strength=np.exp(-((x/.34)**4+(y/.065)**4))
        mouth_strength=np.exp(-((x/.24)**4+((y+.245)/.07)**4))
        strength=np.maximum(eye_strength,mouth_strength)
        valid_front=weights[:,:,0].min(1)>1e-10
        strength*=valid_front
        weights[:,:,0]*=(1+25*strength[:,None])
        weights[:,:,1:]*=(1-.95*strength[:,None,None])
        feature_report=dict(front_preferred_triangles=int((strength>.1).sum()),method='smooth geometric eye/mouth front preference, only previously valid front triangles',geometry_modified=False,gaze_estimated=False,mouth_geometry_corrected=False)
    eye_report=None
    if separate_eyes:
        from eye_components import identify
        component,eye_report=identify(vertices)
        front_eye=(component>0)&(weights[:,:,0].min(1)>1e-10)
        weights[front_eye]=0
        weights[front_eye,:,0]=1
        eye_report['front_only_texture_triangles']=int(front_eye.sum())
        eye_report['other_eye_triangles']='existing source weights retained; no new projection'
    refinement = None
    if refine:
        from refine_texture_boundary import prepare
        fade, fill, refinement = prepare(vertices,weights,coords,images,gains,skin,sample,linear,corner_gains,source_bias)
        if separate_eyes:
            fade[component>0]=1
    grid = math.ceil(math.sqrt(count))
    side = grid * cell
    atlas = np.zeros((side, side, 3), dtype=np.uint8)
    uv = np.empty((count,3,2), dtype=np.float32)
    yy, xx = np.mgrid[:cell,:cell]
    b = np.clip((xx.ravel()+.5-2)/(cell-4),0,1)
    c = np.clip((yy.ravel()+.5-2)/(cell-4),0,1)
    total = np.maximum(b+c,1)
    b, c = b/total, c/total
    bary = np.column_stack((1-b-c,b,c))
    for i in range(count):
        w = bary @ weights[i]
        sums = w.sum(1)
        active = sums >= 1e-6
        color = np.zeros((len(bary),3),dtype=float)
        for j in range(3):
            q = bary @ coords[j,i]
            if np.any(q[:,2] <= 0):
                raise ValueError('Non-positive source depth')
            rgb = sample(images[j], q[:,:2]/q[:,2,None])
            gain=gains[j] if corner_gains is None else bary@corner_gains[j].reshape(-1,3,3)[i]
            corrected=linear(rgb)*gain
            if source_bias is not None:corrected=np.maximum(corrected+source_bias[j],0)
            color += corrected*w[:,j,None]
        color /= np.maximum(sums[:,None],1e-12)
        color = np.where(color <= .0031308, color*12.92, 1.055*np.maximum(color,0)**(1/2.4)-.055)
        # Hidden-region fill is flat: the HTML's view-dependent fill shading is not baked.
        color[~active] = skin
        if refine:
            opacity = (bary @ fade[i])[:,None]
            color = color*opacity + (bary @ fill[i])*(1-opacity)
        y, x = divmod(i,grid)
        atlas[y*cell:(y+1)*cell,x*cell:(x+1)*cell] = np.rint(np.clip(color,0,1)*255).astype(np.uint8).reshape(cell,cell,3)
        uv[i] = np.array([[x*cell+2,y*cell+2],[(x+1)*cell-2,y*cell+2],[x*cell+2,(y+1)*cell-2]])/side
        if i % 1000 == 0:
            print(f'Baking {i}/{count}',flush=True)
    png = io.BytesIO()
    Image.fromarray(atlas).save(png,format='PNG')
    blob = bytearray()
    views, accessors = [], []
    def add(data, target=None):
        while len(blob)%4: blob.append(0)
        view = dict(buffer=0,byteOffset=len(blob),byteLength=len(data))
        if target: view['target']=target
        views.append(view); blob.extend(data)
        return len(views)-1
    for array, kind in ((vertices,'VEC3'),(normals,'VEC3'),(uv.reshape(-1,2),'VEC2')):
        a = np.asarray(array,dtype='<f4')
        acc = dict(bufferView=add(a.tobytes(),34962),componentType=5126,count=len(a),type=kind)
        if len(accessors)==0: acc.update(min=a.min(0).tolist(),max=a.max(0).tolist())
        accessors.append(acc)
    image_view = add(png.getvalue())
    doc = dict(asset={'version':'2.0','generator':'Memento3D optimized workspace bake'},
        extensionsUsed=['KHR_materials_unlit'],scene=0,scenes=[{'nodes':[0]}],
        nodes=[{'mesh':0,'name':'OptimizedHead'}],
        meshes=[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'TEXCOORD_0':2},'material':0,'mode':4}]}],
        materials=[{'name':'OptimizedPhotoTexture','doubleSided':True,'extensions':{'KHR_materials_unlit':{}},'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicFactor':0,'roughnessFactor':1}}],
        textures=[{'source':0,'sampler':0}],samplers=[{'magFilter':9729,'minFilter':9729,'wrapS':33071,'wrapT':33071}],
        images=[{'bufferView':image_view,'mimeType':'image/png'}],
        buffers=[{'byteLength':len(blob)}],bufferViews=views,accessors=accessors)
    if separate_eyes:
        # Shared buffers preserve positions, UVs and normals exactly. Each node
        # references only its own triangles and gets an independent material.
        meshes=[];nodes=[];materials=[]
        for i,name in enumerate(eye_report['object_names']):
            indices=np.flatnonzero(np.repeat(component==i,3)).astype('<u4')
            view=add(indices.tobytes(),34963)
            accessors.append(dict(bufferView=view,componentType=5125,count=len(indices),type='SCALAR'))
            meshes.append({'name':name,'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'TEXCOORD_0':2},'indices':len(accessors)-1,'material':i,'mode':4}]})
            nodes.append({'mesh':i,'name':name})
            material=json.loads(json.dumps(doc['materials'][0]));material['name']=name+'_Photo';materials.append(material)
        doc.update(meshes=meshes,nodes=nodes,materials=materials,scenes=[{'nodes':[0,1,2]}],buffers=[{'byteLength':len(blob)}])
    js=json.dumps(doc,separators=(',',':')).encode()
    js+=b' '*((-len(js))%4)
    blob+=b'\0'*((-len(blob))%4)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:
        f.write(struct.pack('<III',0x46546c67,2,12+8+len(js)+8+len(blob)))
        f.write(struct.pack('<I4s',len(js),b'JSON'));f.write(js)
        f.write(struct.pack('<I4s',len(blob),b'BIN\0'));f.write(blob)
    report=dict(status='GLB_EXPORTED',source=str(source.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),output=str(output.resolve()),triangles=count,atlas_size=side,geometry_modified=False,texture_baked=True,hidden_fill='flat sampled skin; no recovered hidden detail',appearance='optimized default blend and gains; unlit photo material; view-dependent HTML fill shading omitted',units='native model units; physical scale unverified')
    if refine:
        report.update(texture_refinement=refinement,hidden_fill='smooth approximate skin material; not recovered hidden texture')
    if separate_eyes:
        report['eye_components']=eye_report
    if harmonize:
        report.update(color_harmonization=calibration_report,feature_texture=feature_report)
    output.with_suffix('.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cell',type=int,choices=(16,32,48,64),default=32)
    p.add_argument('--refine-texture',action='store_true',help='Experimental V2.1 inward coverage fade and smooth skin fill')
    p.add_argument('--separate-eyes',action='store_true',help='V2.2: split verified existing eyeballs; prefer valid front eye texture')
    p.add_argument('--harmonize-colors',action='store_true',help='Experimental spatial color match and protected front feature texture')
    a=p.parse_args()
    export(a.workspace,a.output,a.cell,a.refine_texture,a.separate_eyes,a.harmonize_colors)
