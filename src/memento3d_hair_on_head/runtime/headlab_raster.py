"""Chunked vectorized barycentric rasterization, deterministic CPU z-buffer.

The integer pixel centers and affine/perspective depth rules match the existing
reference rasterizer. Images are previews; mesh export is independent.
"""
import io
from pathlib import Path
import numpy as np
from PIL import Image
from add_graduation_cap import load
from figurine_preview import accessor


def fragments(screen,faces,width,height,cull=False,max_candidates=1500000):
    triangles=screen[faces];a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    denominator=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    low=np.maximum(np.floor(triangles.min(1)).astype(int),0)
    high=np.minimum(np.ceil(triangles.max(1)).astype(int),[width-1,height-1])
    valid=(denominator<-1e-10) if cull else abs(denominator)>1e-10
    valid&=(low<=high).all(1)
    active=np.flatnonzero(valid);cursor=0
    while cursor<len(active):
        ids=active[cursor:cursor+512];counts=np.prod(high[ids]-low[ids]+1,axis=1)
        if counts.sum()>max_candidates:
            keep=max(1,int(np.searchsorted(np.cumsum(counts),max_candidates)))
            ids=ids[:keep];counts=counts[:keep]
        cursor+=len(ids)
        which=np.repeat(np.arange(len(ids)),counts);face_ids=ids[which]
        offsets=np.arange(int(counts.sum()))-np.repeat(np.r_[0,np.cumsum(counts)[:-1]],counts)
        box_width=high[face_ids,0]-low[face_ids,0]+1
        x=low[face_ids,0]+offsets%box_width;y=low[face_ids,1]+offsets//box_width
        aa=((b[face_ids,1]-c[face_ids,1])*(x-c[face_ids,0])+(c[face_ids,0]-b[face_ids,0])*(y-c[face_ids,1]))/denominator[face_ids]
        bb=((c[face_ids,1]-a[face_ids,1])*(x-c[face_ids,0])+(a[face_ids,0]-c[face_ids,0])*(y-c[face_ids,1]))/denominator[face_ids]
        weights=np.column_stack((aa,bb,1-aa-bb));inside=weights.min(1)>=-1e-8
        yield face_ids[inside],(y*width+x)[inside],weights[inside]


def surface_depth_fast(vertices,faces,camera,size):
    width,height=size;q=np.column_stack((vertices,np.ones(len(vertices))))@camera.T
    positive=(q[:,2]>1e-8);faces=np.asarray(faces)[positive[np.asarray(faces)].all(1)]
    screen=q[:,:2]/np.maximum(q[:,2,None],1e-8);depth=np.full(width*height,np.inf)
    for ids,pixels,weights in fragments(screen,faces,width,height):
        inverse=np.sum(weights/q[faces[ids],2],axis=1);valid=inverse>0
        np.minimum.at(depth,pixels[valid],1/inverse[valid])
    return depth.reshape(height,width)


def render_fast(path,yaw=0,size=300,frame=None,monochrome=False):
    doc,blob=load(Path(path));primitives=[];textures={}
    for i,img in enumerate(doc.get('images',[])):
        view=doc['bufferViews'][img['bufferView']];offset=view.get('byteOffset',0)
        textures[i]=np.asarray(Image.open(io.BytesIO(blob[offset:offset+view['byteLength']])).convert('RGB'),float)/255
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
    zoom=(size-30)/span;canvas=np.tile([.13,.15,.18],(size*size,1));depth=np.full(size*size,-np.inf)
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
            if monochrome:rgb[:]=.68
            nn=np.sum(w[:,:,None]*normal[faces[ids]],axis=1);nn/=np.maximum(np.linalg.norm(nn,axis=1,keepdims=True),1e-8)
            canvas[pixels]=rgb*(.45+.55*np.maximum(nn@light,0))[:,None];depth[pixels]=z
    return Image.fromarray(np.uint8(np.clip(canvas.reshape(size,size,3),0,1)*255))
