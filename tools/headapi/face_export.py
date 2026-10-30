"""Preserve RDF fitted geometry; export color sampling as an explicit approximation."""
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image


def export_face(bundle,crops,slots,out):
    with np.load(bundle,allow_pickle=False) as b:
        vertices=b['vertices'].copy();faces=b['faces'].copy()
        views=b['view_vertices'].copy();cameras=b['cameras'].copy()
    if not np.isfinite(vertices).all():raise ValueError('Nonfinite RDF vertices')
    gray=trimesh.Trimesh(vertices,faces,process=False)
    gray.visual.vertex_colors=[180,180,180,255];gray.export(Path(out)/'rdf_face_gray.glb')
    # Keep geometry exactly. Photographic vertex colors are only a review asset;
    # no albedo recovery, hidden skin completion or eye texture synthesis.
    colors=np.full((len(vertices),4),[180,180,180,255],dtype=np.uint8)
    best=np.full(len(vertices),-np.inf)
    for i,slot in enumerate(slots):
        rgb=np.asarray(Image.open(Path(crops)/(slot+'.png')).convert('RGB'))
        homogeneous=np.column_stack((views[i],np.ones(len(vertices))))@cameras[i].T
        positive=homogeneous[:,2]>1e-8
        uv=np.zeros((len(vertices),2));uv[positive]=homogeneous[positive,:2]/homogeneous[positive,2,None]
        xy=np.rint(uv).astype(int)
        inside=positive&(xy[:,0]>=0)&(xy[:,0]<rgb.shape[1])&(xy[:,1]>=0)&(xy[:,1]<rgb.shape[0])
        # Sparse vertex-depth visibility proxy. Not triangle-perfect occlusion.
        depth=np.full(rgb.shape[:2],np.inf)
        np.minimum.at(depth,(xy[inside,1],xy[inside,0]),homogeneous[inside,2])
        visible=inside.copy();ids=np.flatnonzero(inside)
        visible[ids]=homogeneous[ids,2]<=depth[xy[ids,1],xy[ids,0]]+1e-6
        normals=trimesh.Trimesh(views[i],faces,process=False).vertex_normals
        center=-np.linalg.solve(cameras[i,:,:3],cameras[i,:,3])
        direction=center-views[i];direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
        score=(normals*direction).sum(1)
        # Front is preferred where usable, while profiles can color visible sides.
        score+=.05 if slot=='front' else 0
        valid=visible&(score>0)&(score>best)
        colors[valid,:3]=rgb[xy[valid,1],xy[valid,0]];best[valid]=score[valid]
    photo=trimesh.Trimesh(vertices,faces,process=False,vertex_colors=colors);photo.export(Path(out)/'rdf_face_photo.glb')
    from tools.triposr_probe.audit import render
    preview=photo.copy();preview.vertices=np.asarray(preview.vertices)[:,[2,0,1]]
    render(preview,Path(out)/'rdf_face_photo.png');render(preview,Path(out)/'rdf_face_gray.png',gray=True)
    return {'geometry_preserved_from_bundle':True,'face_identity_verified':False,
            'hair_present':False,'units':'native RDF units; not calibrated millimeters',
            'color_method':'per-vertex projection with sparse visibility proxy; not recovered albedo',
            'color_unobserved_fill':'neutral gray','views_used':list(slots),'ready_for_machine':False}


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--crops',type=Path,required=True)
    a=p.parse_args();report=json.loads((a.fit/'report.json').read_text())
    from tools.headapi.service import sha,write
    if sha(a.fit/'projection_bundle.npz')!=report['projection_bundle_sha256']:raise ValueError('Fit bundle changed')
    result=export_face(a.fit/'projection_bundle.npz',a.crops,report['views_used'],a.fit)
    write(a.fit/'export.report.json',result)
