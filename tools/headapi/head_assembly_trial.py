"""Offline adapter: split a pinned template volume and assemble a complete head.

This is a visual trial, not an automatic semantic segmentation or a mesh union.
"""
import argparse,json,sys,pickle
from pathlib import Path
import numpy as np
import trimesh
from .service import sha,write

def removal_mask(vertices,faces,minimum,maximum):
    """Remove only triangles wholly inside the explicitly annotated head volume."""
    vertices=np.asarray(vertices);faces=np.asarray(faces);lo=np.asarray(minimum);hi=np.asarray(maximum)
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or (hi<=lo).any():raise ValueError('Invalid head volume')
    inside=((vertices>=lo)&(vertices<=hi)).all(1)
    return inside[faces].all(1)

def condition_neck(vertices,neck_mask):
    """Constrain only semantic neck vertices; face/scalp/ears are fixed."""
    result=np.array(vertices,dtype=float,copy=True);mask=np.asarray(neck_mask,dtype=bool)
    if mask.shape!=(len(result),) or not np.isfinite(result).all():raise ValueError('Invalid semantic neck input')
    center=np.array([-.022,.080]);radii=np.array([.035,.045]);offset=result[:,:2]-center
    radius=np.sqrt(((offset/radii)**2).sum(1));factor=np.minimum(1,1/np.maximum(radius,1e-12))
    w=np.clip((.81-result[:,2])/.015,0,1);w=w*w*(3-2*w);w*=mask
    result[:,:2]+=w[:,None]*(offset*factor[:,None]-offset)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--alignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--selection-policy',choices=['box_v1','narrow_v2','semantic_neck_v3'],default='semantic_neck_v3');a=p.parse_args()
    a.runtime=a.runtime.resolve();a.package=a.package.resolve();a.output=a.output.resolve()
    if a.output.exists():raise ValueError('Choose a fresh attempt directory')
    template=a.runtime/'GLB/men.glb';head_path=a.package/'input04/head.glb';manifest=a.package/'manifest.json'
    expected=json.loads(manifest.read_text(encoding='utf-8'))['cases']['input04']['sha256']['head.glb']
    if sha(head_path)!=expected:raise ValueError('Head package pin changed')
    if sha(template)!='96496c2d10e2896a780fb060c2171773000df22a8649e15d66096d26080db84d':raise ValueError('Template pin changed')
    alignment=json.loads(a.alignment.read_text(encoding='utf-8'));matrix=np.array(alignment['alignment_matrix_working'])
    source=trimesh.load(template,force='mesh',process=False);head=trimesh.load(head_path,force='mesh',process=False)
    axes=np.eye(4);axes[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
    extra=[]
    if a.selection_policy=='semantic_neck_v3':
        root=Path(__file__).resolve().parents[2];bundle=root/'runs/experiments/api-demo-uv-01/case1/rdf/projection_bundle.npz';masks=root/'vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl'
        if sha(masks)!='ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492':raise ValueError('Semantic mask pin changed')
        with np.load(bundle,allow_pickle=False) as raw:raw_vertices=raw['vertices'];raw_faces=raw['faces']
        if not np.array_equal(head.vertices.astype(np.float32),raw_vertices[raw_faces].reshape(-1,3).astype(np.float32)):raise ValueError('Head topology/order differs from semantic mask binding')
        with masks.open('rb') as stream:neck_ids=pickle.load(stream,encoding='latin1')['neck']
        neck_mask=np.isin(raw_faces.ravel(),neck_ids);extra=[bundle,masks]
    source.apply_transform(axes);head.apply_transform(matrix@axes)
    if a.selection_policy=='semantic_neck_v3':
        original=head.vertices.copy();head.vertices=condition_neck(head.vertices,neck_mask)
        if not np.array_equal(original[~neck_mask],head.vertices[~neck_mask]):raise ValueError('Non-neck geometry changed')
    if a.selection_policy!='box_v1':
        hv,hf,hu=trimesh.intersections.slice_faces_plane(head.vertices,head.faces,[0,0,1],[0,0,.765],uv=head.visual.uv)
        head=trimesh.Trimesh(hv,hf,visual=trimesh.visual.TextureVisuals(uv=hu,material=head.visual.material),process=False)
    a.output.mkdir(parents=True)
    # Native asset units are inherited design coordinates, not calibrated measurements.
    report={'policy':'complete_head_template_assembly_trial_v3','selection_policy':a.selection_policy,'status':'CANDIDATES_EXPORTED_NEEDS_VISUAL_REVIEW','input_sha256':{str(q):sha(q) for q in [template,head_path,manifest,a.alignment]+extra},'code_sha256':sha(Path(__file__)),'alignment_matrix_working':matrix.tolist(),'source_axes':'glTF Y-up to working Z-up, front -Y','landmark_residuals':alignment['landmark_residuals'],'semantic_segmentation':False,'semantic_neck_conditioning':a.selection_policy=='semantic_neck_v3','non_neck_vertices_fixed':True,'neck_base_trim_z_working':.765,'geometry_union_performed':False,'fresh_reconstruction':False,'api_changed':False,'ready_for_machine':False,'cases':{}}
    cases=[('collar-low',.745),('collar-high',.765)] if a.selection_policy=='box_v1' else [('neck-low',.775),('neck-high',.785)]
    for name,height in cases:
        folder=a.output/name;folder.mkdir()
        lower=[-.20,-.14,height] if a.selection_policy=='box_v1' else [-.12,-.09,height]
        upper=[.17,.24,1.10] if a.selection_policy=='box_v1' else [.09,.19,1.10]
        remove=removal_mask(source.vertices,source.faces,lower,upper)
        if a.selection_policy!='box_v1':remove|=removal_mask(source.vertices,source.faces,[-.15,-.06,.94],[.13,.20,1.1])
        if not remove.any() or remove.all():raise ValueError('Invalid head selection')
        body=source.submesh([np.flatnonzero(~remove)],append=True,repair=False)
        old=source.submesh([np.flatnonzero(remove)],append=True,repair=False)
        expected_corners=source.vertices[source.faces[~remove]]
        if not np.array_equal(body.vertices[body.faces],expected_corners):raise ValueError('Retained body geometry changed')
        expected_uv=source.visual.uv[source.faces[~remove]]
        if not np.array_equal(body.visual.uv[body.faces],expected_uv):raise ValueError('Retained body UV changed')
        scene=trimesh.Scene();scene.add_geometry(body,node_name='Body_Template',geom_name='Body_Template');scene.add_geometry(head,node_name='Personalized_Complete_Head',geom_name='Personalized_Complete_Head')
        # glTF world is Y-up; retain the working Z-up frame only in the report.
        export_axes=np.linalg.inv(axes);scene.apply_transform(export_axes)
        body_scene=trimesh.Scene(body);body_scene.apply_transform(export_axes)
        old_scene=trimesh.Scene(old);old_scene.apply_transform(export_axes)
        scene.export(folder/'assembly.glb');body_scene.export(folder/'body.glb');old_scene.export(folder/'removed-template-head.glb')
        reread=trimesh.load(folder/'assembly.glb',force='scene',process=False)
        if len(reread.geometry)!=2:raise ValueError('Roundtrip did not retain separate body/head components')
        report['cases'][name]={'volume_min':lower,'volume_max':upper,'removed_triangles':int(remove.sum()),'retained_triangles':int((~remove).sum()),'head_triangles':len(head.faces),'body_geometry_unchanged_before_export':True,'body_uv_unchanged_before_export':True,'assembled_sha256':sha(folder/'assembly.glb'),'separate_components_after_roundtrip':2,'neck_join':'open/interpenetrating preview only; not stitched or capped','head_bounds_working':head.bounds.tolist()}
        print(name,int(remove.sum()),'removed',flush=True)
    write(a.output/'report.json',report)
    print(report['status'],flush=True)

if __name__=='__main__':main()
