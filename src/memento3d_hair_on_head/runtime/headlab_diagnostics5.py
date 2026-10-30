"""Spatially located geometry screening; no minimum-wall certificate."""
import json
import numpy as np
import trimesh
from headlab_printcheck import screen
from run_headlab_v4 import save_gray
from figurine_preview import add_mesh
from add_graduation_cap import save


def chord_map(mesh,samples,threshold):
    areas=mesh.area_faces;quantiles=(np.arange(samples)+.5)/samples*areas.sum()
    ids=np.unique(np.searchsorted(np.cumsum(areas),quantiles));triangles=mesh.triangles
    a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    records=[];eps=max(float(mesh.extents.max())*1e-7,1e-8)
    for face_id in ids:
        start=mesh.triangles_center[face_id];direction=-mesh.face_normals[face_id];origin=start+direction*eps
        h=np.cross(direction,e2);det=np.einsum('ij,ij->i',e1,h);valid=abs(det)>1e-12
        inv=np.divide(1,det,out=np.zeros_like(det),where=valid);s=origin-a
        u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1);v=q@direction*inv;distance=np.einsum('ij,ij->i',e2,q)*inv
        hit=valid&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(distance>eps)
        hit[face_id]=False
        length=float(distance[hit].min()+eps) if hit.any() else None
        records.append({'face_id':int(face_id),'point_stl_mm':start.tolist(),'inward_normal':direction.tolist(),'sampled_chord_mm':length,'below_trial_threshold':length is not None and length<threshold})
    return records


def diagnose(stl,output,profile):
    report=screen(stl,profile);mesh=trimesh.load(stl,force='mesh')
    records=chord_map(mesh,profile['ray_samples'],profile['trial_feature_mm'])
    (output/'chord_samples.json').write_text(json.dumps({'method':'area-stratified sampled inward-normal ray chords','minimum_wall_thickness_verified':False,'samples':records},indent=2))
    colors=np.tile([.65,.68,.7,1],(len(mesh.vertices),1));transform=np.eye(4)
    # STL Z-up -> Y-up GLB, keeping coordinates in mm before add_mesh conversion.
    transform[:3,:3]=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    base=mesh.copy();base.apply_transform(transform)
    doc={'asset':{'version':'2.0'},'scenes':[{'nodes':[]}],'scene':0,'nodes':[],'meshes':[],'materials':[],'accessors':[],'bufferViews':[]};blob=bytearray()
    add_mesh(doc,blob,'Head',base,colors)
    flagged=[r for r in records if r['below_trial_threshold']]
    if flagged:
        points=trimesh.transform_points(np.asarray([r['point_stl_mm'] for r in flagged]),transform)
        spheres=[]
        for point in points:
            ball=trimesh.creation.icosphere(subdivisions=1,radius=.12);ball.apply_translation(point);spheres.append(ball)
        markers=trimesh.util.concatenate(spheres);add_mesh(doc,blob,'ScreeningMarkers_NOT_PrintParts',markers,np.tile([1,.15,.02,1],(len(markers.vertices),1)))
    save(output/'screening_map.glb',doc,blob)
    report['located_screening']={'sampling':'area-stratified face centers','flagged_samples':len(flagged),'file':'chord_samples.json','map':'screening_map.glb','diagnostic_markers_not_exported_to_print_stl':True}
    return report
