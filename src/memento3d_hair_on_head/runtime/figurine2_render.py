"""Linear-light preview for V2 GLB vertex colors; approximation, not calibrated rendering."""
import io
from pathlib import Path
import numpy as np
from PIL import Image
from add_graduation_cap import load
from figurine_preview import accessor
from headlab_raster import fragments

def srgb_to_linear(x):
    x=np.asarray(x,float)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def linear_to_srgb(x):
    x=np.maximum(np.asarray(x,float),0)
    return np.where(x<=.0031308,x*12.92,1.055*x**(1/2.4)-.055)

def render_fast(path,yaw=0,size=300,frame=None,monochrome=False):
    doc,blob=load(Path(path));primitives=[];textures={}
    for i,img in enumerate(doc.get('images',[])):
        view=doc['bufferViews'][img['bufferView']];offset=view.get('byteOffset',0)
        textures[i]=srgb_to_linear(np.asarray(Image.open(io.BytesIO(blob[offset:offset+view['byteLength']])).convert('RGB'),float)/255)
    for node_id in doc['scenes'][0]['nodes']:
        for primitive in doc['meshes'][doc['nodes'][node_id]['mesh']]['primitives']:
            a=primitive['attributes'];p=accessor(doc,blob,a['POSITION'])*1000;n=accessor(doc,blob,a['NORMAL'])
            f=accessor(doc,blob,primitive['indices']).reshape(-1,3).astype(int) if 'indices' in primitive else np.arange(len(p)).reshape(-1,3)
            color=accessor(doc,blob,a['COLOR_0'])[:,:3] if 'COLOR_0' in a else np.ones_like(p)
            material=doc['materials'][primitive['material']].get('pbrMetallicRoughness',{});texture=None;uv=None
            if 'baseColorTexture' in material:
                texture=textures[doc['textures'][material['baseColorTexture']['index']]['source']];uv=accessor(doc,blob,a['TEXCOORD_0'])
            else:color*=np.asarray(material.get('baseColorFactor',[1,1,1,1]))[:3]
            primitives.append((p,n,f,color,texture,uv))
    angle=np.deg2rad(yaw);rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    all_points=np.vstack([x[0] for x in primitives]);center=(all_points.min(0)+all_points.max(0))/2;span=np.ptp(all_points,axis=0).max()*1.15
    if frame is not None:center=np.asarray(frame[0]);span=frame[1]
    zoom=(size-30)/span;canvas=np.tile(srgb_to_linear(np.array([.13,.15,.18])),(size*size,1));depth=np.full(size*size,-np.inf)
    light=np.array([-.3,.55,1.]);light/=np.linalg.norm(light)
    for p,n,faces,color,texture,uv in primitives:
        world=(p-center)@rotation.T;normal=n@rotation.T
        screen=np.column_stack((size/2+world[:,0]*zoom,size/2-world[:,1]*zoom))
        for ids,pixels,w in fragments(screen,faces,size,size,cull=True):
            z=np.sum(w*world[faces[ids],2],axis=1)
            keep=z>depth[pixels];ids,pixels,w,z=ids[keep],pixels[keep],w[keep],z[keep]
            if not len(ids):continue
            # Select one nearest fragment per pixel inside the chunk.
            order=np.lexsort((z,pixels));ordered_pixels=pixels[order]
            winners=order[np.r_[ordered_pixels[1:]!=ordered_pixels[:-1],True]]
            ids,pixels,w,z=ids[winners],pixels[winners],w[winners],z[winners]
            rgb=np.sum(w[:,:,None]*color[faces[ids]],axis=1)
            if texture is not None:
                st=np.clip(np.sum(w[:,:,None]*uv[faces[ids]],axis=1),0,1)
                xy=(st*np.array([texture.shape[1]-1,texture.shape[0]-1])).astype(int);rgb*=texture[xy[:,1],xy[:,0]]
            if monochrome:rgb[:]=srgb_to_linear(np.array(.68))
            nn=np.sum(w[:,:,None]*normal[faces[ids]],axis=1);nn/=np.maximum(np.linalg.norm(nn,axis=1,keepdims=True),1e-8)
            canvas[pixels]=rgb*(.45+.55*np.maximum(nn@light,0))[:,None];depth[pixels]=z
    return Image.fromarray(np.uint8(np.clip(linear_to_srgb(canvas.reshape(size,size,3)),0,1)*255))
