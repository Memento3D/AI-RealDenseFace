"""Original procedural mortarboard asset and experimental fit to a V2 head GLB."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def load(path):
    raw=path.read_bytes()
    if struct.unpack('<II',raw[:8])!=(0x46546c67,2):raise ValueError('Expected GLB 2')
    n=struct.unpack('<I',raw[12:16])[0]
    doc=json.loads(raw[20:20+n]);start=28+n
    return doc,bytearray(raw[start:start+doc['buffers'][0]['byteLength']])


def save(path,doc,blob):
    if path.exists():raise ValueError('Output already exists: '+str(path))
    doc['buffers']=[{'byteLength':len(blob)}]
    js=json.dumps(doc,separators=(',',':')).encode();js+=b' '*((-len(js))%4)
    blob=bytes(blob)+b'\0'*((-len(blob))%4)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(struct.pack('<III',0x46546c67,2,28+len(js)+len(blob)))
        f.write(struct.pack('<I4s',len(js),b'JSON'));f.write(js)
        f.write(struct.pack('<I4s',len(blob),b'BIN\0'));f.write(blob)


def append(doc,blob,name,triangles,color):
    p=np.asarray(triangles,dtype='<f4').reshape(-1,3)
    t=p.reshape(-1,3,3);n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);length=np.linalg.norm(n,axis=1)
    if np.any(length<1e-12):raise ValueError('Degenerate cap triangle')
    n=np.repeat(n/length[:,None],3,axis=0).astype('<f4')
    if name in ('GraduationCap_Body','Hair_ShortTemplate'):
        _,inverse=np.unique(p,axis=0,return_inverse=True)
        summed=np.stack([np.bincount(inverse,weights=n[:,k]) for k in range(3)],axis=1)
        summed/=np.maximum(np.linalg.norm(summed,axis=1)[:,None],1e-12)
        n=summed[inverse].astype('<f4')
    ids=[]
    for array in (p,n):
        while len(blob)%4:blob.append(0)
        view=dict(buffer=0,byteOffset=len(blob),byteLength=array.nbytes,target=34962)
        doc.setdefault('bufferViews',[]).append(view);blob.extend(array.tobytes())
        a=dict(bufferView=len(doc['bufferViews'])-1,componentType=5126,count=len(array),type='VEC3')
        if not ids:a.update(min=p.min(0).tolist(),max=p.max(0).tolist())
        doc.setdefault('accessors',[]).append(a);ids.append(len(doc['accessors'])-1)
    material=dict(name=name+'_Material',doubleSided=True,pbrMetallicRoughness=dict(baseColorFactor=[*color,1],metallicFactor=0,roughnessFactor=.8))
    doc.setdefault('materials',[]).append(material)
    primitive=dict(attributes={'POSITION':ids[0],'NORMAL':ids[1]},material=len(doc['materials'])-1,mode=4)
    doc.setdefault('meshes',[]).append(dict(name=name,primitives=[primitive]))
    doc.setdefault('nodes',[]).append(dict(name=name,mesh=len(doc['meshes'])-1))
    doc['scenes'][doc.get('scene',0)]['nodes'].append(len(doc['nodes'])-1)


def bridge(rings,close_ends=False):
    tri=[];rings=np.asarray(rings);count=rings.shape[1]
    for j in range(len(rings)-1):
        for i in range(count):
            k=(i+1)%count;a,b,c,d=rings[j,i],rings[j,k],rings[j+1,k],rings[j+1,i]
            tri.extend([[a,b,c],[a,c,d]])
    if close_ends:
        for ring,reverse in ((rings[0],True),(rings[-1],False)):
            center=ring.mean(0)
            for i in range(count):
                face=[center,ring[i],ring[(i+1)%count]]
                tri.append(face[::-1] if reverse else face)
    return np.asarray(tri)


def tube(points,radius,segments=12):
    points=np.asarray(points);rings=[]
    for i,p in enumerate(points):
        direction=points[min(i+1,len(points)-1)]-points[max(i-1,0)]
        direction/=np.linalg.norm(direction)
        helper=np.array([0.,1,0]) if abs(direction[1])<.9 else np.array([1.,0,0])
        x=np.cross(direction,helper);x/=np.linalg.norm(x);y=np.cross(direction,x)
        angles=np.arange(segments)*2*np.pi/segments
        rings.append(p+radius*(np.cos(angles)[:,None]*x+np.sin(angles)[:,None]*y))
    return bridge(rings,True)


def design(head):
    # Fit heuristically in the existing canonical FLAME axes (+Y up). Native
    # units remain unchanged; no physical-size or universal-fit claim is made.
    low,high=head.min(0),head.max(0);height=high[1]-low[1]
    top=high[1];bottom=top-.145*height
    band=head[(head[:,1]>bottom-.02*height)&(head[:,1]<bottom+.02*height)]
    if len(band)<20:raise ValueError('Not enough scalp samples to fit the cap')
    center=(band.min(0)+band.max(0))/2;cx,cz=center[0],center[2]
    base_x=np.ptp(band[:,0])*.5;base_z=np.ptp(band[:,2])*.5
    gap=.012*height
    levels=np.linspace(bottom,top+.012*height,9);outer=[];angles=np.arange(96)*2*np.pi/96
    for y in levels:
        ratio=(y-bottom)/(levels[-1]-bottom)
        rx=base_x*(1-.40*ratio);rz=base_z*(1-.40*ratio)
        section=head[abs(head[:,1]-y)<.028*height]
        if len(section):
            distance=np.sqrt(((section[:,0]-cx)/rx)**2+((section[:,2]-cz)/rz)**2)
            factor=max(1,float(distance.max()))
            rx*=factor;rz*=factor
        outer.append(np.column_stack((cx+(rx+gap)*np.cos(angles),np.full(len(angles),y),cz+(rz+gap)*np.sin(angles))))
    inner=[ring.copy() for ring in outer]
    for ring in inner:
        ring[:,0]=cx+(ring[:,0]-cx)*.97;ring[:,2]=cz+(ring[:,2]-cz)*.97
    shell=bridge(outer)
    shell=np.concatenate((shell,bridge(inner)[:,::-1],bridge([outer[0],inner[0]]),bridge([outer[-1],inner[-1]])))
    board_y=levels[-1];half=max(base_x,base_z)*1.35;thickness=.018*height
    corners=np.array([[-half,-half],[half,-half],[half,half],[-half,half]])
    a=np.pi/4;rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    corners=corners@rot.T
    board=bridge([np.column_stack((cx+corners[:,0],np.full(4,y),cz+corners[:,1])) for y in (board_y,board_y+thickness)],True)
    y=board_y+thickness+.006*height
    button=tube([[cx,y-.003*height,cz],[cx,y+.008*height,cz]],.018*height,24)
    edge=half*1.25
    cord=tube([[cx,y,cz],[cx+edge*.4,y+.004*height,cz+.02*height],[cx+edge*.8,y,cz+.025*height],[cx+edge,y-.04*height,cz+.025*height],[cx+edge,y-.23*height,cz+.025*height]],.004*height)
    tassel=tube([[cx+edge,y-.23*height,cz+.025*height],[cx+edge,y-.31*height,cz+.025*height]],.015*height,20)
    return [('GraduationCap_Body',shell,[.018,.022,.035]),('GraduationCap_Board',board,[.025,.03,.045]),
        ('GraduationCap_Button',button,[.15,.08,.01]),('GraduationCap_Cord',cord,[.65,.38,.055]),('GraduationCap_Tassel',tassel,[.65,.38,.055])],dict(fit='heuristic scalp envelope in canonical FLAME axes',physical_units_verified=False,collision_verified=False,top=float(top),bottom=float(bottom))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists():raise ValueError('Choose a new --output-dir')
    doc,blob=load(args.head)
    # Supported exporter stores head positions in accessor zero without transforms.
    if doc['nodes'][0].get('name') not in ('Head','OptimizedHead'):raise ValueError('Expected Memento3D canonical head GLB')
    if any(k in doc['nodes'][0] for k in ('matrix','translation','rotation','scale')):raise ValueError('Transformed head is unsupported')
    a=doc['accessors'][0];v=doc['bufferViews'][a['bufferView']]
    if a['componentType']!=5126 or a['type']!='VEC3' or 'byteStride' in v:raise ValueError('Unsupported positions')
    head=np.frombuffer(blob,dtype='<f4',count=a['count']*3,offset=v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,3).copy()
    if not np.isfinite(head).all():raise ValueError('Non-finite head')
    parts,fit=design(np.unique(head,axis=0));out.mkdir(parents=True,exist_ok=False)
    cap=dict(asset={'version':'2.0','generator':'Memento3D original procedural cap 01'},scene=0,scenes=[{'nodes':[]}]);capblob=bytearray()
    for name,triangles,color in parts:
        append(cap,capblob,name,triangles,color)
        append(doc,blob,name,triangles,color)
    save(out/'graduation_cap_01.glb',cap,capblob)
    save(out/'head.with_cap.glb',doc,blob)
    report=dict(status='CAP_ASSEMBLY_EXPORTED',asset_origin='original procedural design',source_head_sha256=hashlib.sha256(args.head.read_bytes()).hexdigest(),head_geometry_modified=False,hair_added=False,fit=fit,printability_verified=False,objects=[x[0] for x in parts])
    (out/'cap.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('Output: '+str(out))


if __name__=='__main__':main()
