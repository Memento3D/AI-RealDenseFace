"""GLB assembly and CPU review images; no network required to view the report."""
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from add_graduation_cap import load, save
TEXTURED_EYE_SUPPORT=True


def accessor(doc, blob, index):
    a=doc['accessors'][index]; view=doc['bufferViews'][a['bufferView']]
    dtype={5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']]
    channels={'VEC3':3,'VEC2':2,'VEC4':4,'SCALAR':1}[a['type']]
    if 'byteStride' in view: raise ValueError('Interleaved GLB unsupported')
    return np.frombuffer(blob,dtype=dtype,count=a['count']*channels,
                         offset=view.get('byteOffset',0)+a.get('byteOffset',0)).reshape(a['count'],channels).copy()


def add_mesh(doc, blob, name, mesh, colors):
    ids=[]
    arrays=[np.asarray(mesh.vertices/1000,dtype='<f4'),np.asarray(mesh.metadata.get('eye_normals',mesh.vertex_normals),dtype='<f4'),np.asarray(colors,dtype='<f4'),np.asarray(mesh.faces,dtype='<u4').reshape(-1,1)]
    for array, kind, component in zip(arrays,['VEC3','VEC3','VEC4','SCALAR'],[5126,5126,5126,5125]):
        while len(blob)%4: blob.append(0)
        doc['bufferViews'].append(dict(buffer=0,byteOffset=len(blob),byteLength=array.nbytes))
        blob.extend(array.tobytes())
        a=dict(bufferView=len(doc['bufferViews'])-1,componentType=component,count=len(array),type=kind)
        if not ids: a.update(min=array.min(0).tolist(),max=array.max(0).tolist())
        doc['accessors'].append(a); ids.append(len(doc['accessors'])-1)
    pb=dict(metallicFactor=0,roughnessFactor=.72)
    uv_id=None
    if 'eye_texture_png' in mesh.metadata:
        uv=np.asarray(mesh.metadata['eye_uv'],dtype='<f4')
        while len(blob)%4:blob.append(0)
        doc['bufferViews'].append(dict(buffer=0,byteOffset=len(blob),byteLength=uv.nbytes));blob.extend(uv.tobytes())
        doc['accessors'].append(dict(bufferView=len(doc['bufferViews'])-1,componentType=5126,count=len(uv),type='VEC2'));uv_id=len(doc['accessors'])-1
        while len(blob)%4:blob.append(0)
        raw=mesh.metadata['eye_texture_png'];doc['bufferViews'].append(dict(buffer=0,byteOffset=len(blob),byteLength=len(raw)));blob.extend(raw)
        doc.setdefault('images',[]).append(dict(bufferView=len(doc['bufferViews'])-1,mimeType='image/png'))
        doc.setdefault('textures',[]).append(dict(source=len(doc['images'])-1))
        pb['baseColorTexture']={'index':len(doc['textures'])-1}
    doc['materials'].append(dict(name=name,doubleSided=True,pbrMetallicRoughness=pb))
    doc['meshes'].append(dict(name=name,primitives=[dict(attributes=dict(POSITION=ids[0],NORMAL=ids[1],COLOR_0=ids[2]),indices=ids[3],material=len(doc['materials'])-1)]))
    if uv_id is not None:doc['meshes'][-1]['primitives'][0]['attributes']['TEXCOORD_0']=uv_id
    doc['nodes'].append(dict(name=name,mesh=len(doc['meshes'])-1))
    doc['scenes'][0]['nodes'].append(len(doc['nodes'])-1)


def assemble(source, destination, positions_mm, normals, additions, hide_eyes=False):
    doc,blob=load(source)
    for index, array in [(0,positions_mm/1000),(1,normals)]:
        a=doc['accessors'][index];view=doc['bufferViews'][a['bufferView']]
        offset=view.get('byteOffset',0)+a.get('byteOffset',0)
        raw=np.asarray(array,dtype='<f4').tobytes()
        if len(raw)!=a['count']*12: raise ValueError('Position layout differs from source exporter')
        blob[offset:offset+len(raw)]=raw
        if index==0:a.update(min=array.min(0).tolist(),max=array.max(0).tolist())
    if hide_eyes:
        doc['scenes'][0]['nodes']=[i for i in doc['scenes'][0]['nodes'] if not doc['nodes'][i].get('name','').startswith('Eye_')]
    for name,mesh,colors in additions: add_mesh(doc,blob,name,mesh,colors)
    doc.setdefault('asset',{})['extras']={'length_unit':'meter','prototype':'Memento figurine V3; source-derived face, original procedural details'}
    save(destination,doc,blob)


def render(path, yaw=0, size=420, monochrome=False, frame=None):
    doc,blob=load(path);textures={}
    for i,img in enumerate(doc.get('images',[])):
        v=doc['bufferViews'][img['bufferView']];raw=blob[v.get('byteOffset',0):v.get('byteOffset',0)+v['byteLength']]
        textures[i]=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'),float)/255
    primitives=[]
    for nodeid in doc['scenes'][0]['nodes']:
        for p in doc['meshes'][doc['nodes'][nodeid]['mesh']]['primitives']:
            a=p['attributes'];xyz=accessor(doc,blob,a['POSITION'])*1000
            normal=accessor(doc,blob,a['NORMAL'])
            faces=accessor(doc,blob,p['indices']).reshape(-1,3).astype(int) if 'indices'in p else np.arange(len(xyz)).reshape(-1,3)
            colors=accessor(doc,blob,a['COLOR_0'])[:,:3] if 'COLOR_0'in a else np.ones_like(xyz)
            mat=doc['materials'][p['material']];pb=mat.get('pbrMetallicRoughness',{})
            tex=None;uv=None
            if 'baseColorTexture'in pb:
                tex=textures[doc['textures'][pb['baseColorTexture']['index']]['source']];uv=accessor(doc,blob,a['TEXCOORD_0'])
            else:colors*=np.asarray(pb.get('baseColorFactor',[1,1,1,1]))[:3]
            primitives.append((xyz,normal,faces,colors,tex,uv))
    angle=np.deg2rad(yaw);rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    allv=np.vstack([p[0] for p in primitives]);center=(allv.min(0)+allv.max(0))/2
    span=max(np.ptp(allv,axis=0))*1.15
    if frame is not None:center=np.asarray(frame[0],float);span=float(frame[1])
    zoom=(size-30)/span
    canvas=np.full((size,size,3),[.13,.15,.18]);depth=np.full((size,size),-np.inf)
    light=np.array([-.3,.55,1.]);light/=np.linalg.norm(light)
    for xyz,normals,faces,colors,tex,uv in primitives:
        world=(xyz-center)@rot.T; normals=normals@rot.T
        screen=np.column_stack((size/2+world[:,0]*zoom,size/2-world[:,1]*zoom))
        triangles=screen[faces]
        lows=np.maximum(np.ceil(triangles.min(1)).astype(int),0)
        highs=np.minimum(np.floor(triangles.max(1)).astype(int),size-1)
        a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
        dens=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        usable=(dens < -1e-10)&(lows<=highs).all(1)
        for f,tri,lo,hi,den in zip(faces[usable],triangles[usable],lows[usable],highs[usable],dens[usable]):
            a,b,c=tri
            yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
            aa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
            bb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
            weights=np.stack((aa,bb,1-aa-bb),-1);z=weights@world[f,2]
            old=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
            keep=(weights.min(-1)>=-1e-6)&(z>old)
            if not keep.any():continue
            w=weights[keep];rgb=w@colors[f]
            if tex is not None:
                st=np.clip(w@uv[f],0,1);xy=(st*np.array([tex.shape[1]-1,tex.shape[0]-1])).astype(int);rgb*=tex[xy[:,1],xy[:,0]]
            if monochrome:rgb=np.full_like(rgb,.68)
            nn=w@normals[f];nn/=np.maximum(np.linalg.norm(nn,axis=1,keepdims=True),1e-8)
            shade=.45+.55*np.maximum(nn@light,0)
            canvas[lo[1]:hi[1]+1,lo[0]:hi[0]+1][keep]=rgb*shade[:,None];old[keep]=z[keep]
    return Image.fromarray(np.uint8(np.clip(canvas,0,1)*255))


def review(output, variants, report):
    rows=[]
    for name,path in variants:
        row=Image.new('RGB',(420*3,450),(22,26,32));ImageDraw.Draw(row).text((12,8),name,fill='white')
        for j,yaw in enumerate([0,65,180]):row.paste(render(path,yaw), (420*j,30))
        filename=name+'.png';row.save(output/filename);rows.append((name,filename))
    sheet=Image.new('RGB',(1260,450*len(rows)),(22,26,32))
    for j,(_,filename) in enumerate(rows):sheet.paste(Image.open(output/filename),(0,j*450))
    sheet.save(output/'comparison.png')
    # Static report stays usable offline; GLBs open in Blender/3D Viewer.
    body=''.join('<h2>'+name+'</h2><img width="1260" src="'+filename+'"><p><a href="'+name+'.glb">GLB: '+name+'</a></p>' for name,filename in rows)
    body='<h2>Hair mask proposals — review before accepting the hairstyle</h2>'+''.join('<img width="400" src="'+v+'.hair_overlay.png">' for v in ['front','left','right'])+body
    if (output/'front.hairline_fit.png').is_file():
        body='<h2>Detected hairline check</h2><p>Green: detected hair. Cyan: detected hair/skin interface. Red: projected supported 3D edge. Orange: interpolated edge without direct observed support. The hair_observation_support model colors unsupported hair orange; this is not a detected region.</p>'+''.join('<img width="512" src="'+v+'.hairline_fit.png">' for v in ['front','left','right'])+body
    html='<!doctype html><meta charset="utf-8"><title>Memento Figurine V3</title><style>body{background:#161a20;color:#eee;font:16px system-ui;margin:28px}img{max-width:100%;border-radius:12px}pre{white-space:pre-wrap}</style><h1>Personalized printable figurine — feasibility review</h1><p>Source-expression head preserved. Hair is a solid photo-constrained design. Smile is a synthetic customization preview. Colors are painting guides; the resin printer does not print these colors.</p><p>STL units: mm. GLB units: meters. Physical print, supports, resin tolerance and slicer checks remain unverified.</p>'+body+'<h2>Geometry report</h2><pre>'+json.dumps(report,indent=2).replace('&','&amp;').replace('<','&lt;')+'</pre>'
    (output/'review.html').write_text(html,encoding='utf-8')


def front_surface_sampler(triangles):
    """Orthographic +Z depth sampler restricted to the mouth's existing surface."""
    points=np.asarray(triangles); faces=points.reshape(-1,3,3)
    low=points[:,:2].min(0)-.5;high=points[:,:2].max(0)+.5
    step=.045;shape=np.ceil((high-low)/step).astype(int)+1
    depth=np.full((shape[1],shape[0]),-np.inf)
    for tri3 in faces:
        tri=(tri3[:,:2]-low)/step
        lo=np.maximum(np.floor(tri.min(0)).astype(int),0);hi=np.minimum(np.ceil(tri.max(0)).astype(int),shape-1)
        if (lo>hi).any():continue
        a,b,c=tri;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-10:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        aa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        bb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        cc=1-aa-bb;inside=(aa>=0)&(bb>=0)&(cc>=0)
        z=aa*tri3[0,2]+bb*tri3[1,2]+cc*tri3[2,2]
        old=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1];np.maximum(old,np.where(inside,z,-np.inf),out=old)
    def sample(x,y):
        i,j=np.rint((np.array([x,y])-low)/step).astype(int)
        if not (0<=i<shape[0] and 0<=j<shape[1]) or not np.isfinite(depth[j,i]):raise ValueError('Mouth relief has no front surface support')
        return float(depth[j,i])
    return sample
