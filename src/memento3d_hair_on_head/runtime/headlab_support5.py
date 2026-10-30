"""Automatic vertical supports and coarse voxel layer precheck, not machine slicing.

No exposure, peel-force or resin calibration is inferred. This stage is an
experimental manufacturing preparation artifact, never a machine-ready file.
"""
import json
import numpy as np
import trimesh
from scipy import ndimage as ndi


def candidate_orientations(mesh):
    rows=[];areas=mesh.area_faces;normals=mesh.face_normals
    for tilt in [20,35,50]:
        for yaw in [0,180]:
            rx=trimesh.transformations.rotation_matrix(np.deg2rad(tilt),[1,0,0])
            rz=trimesh.transformations.rotation_matrix(np.deg2rad(yaw),[0,0,1])
            rotation=rx@rz  # yaw before tilt; unlike the V4 z-area invariant search.
            transformed=normals@rotation[:3,:3].T
            front=np.clip(-normals[:,1],0,1)
            cost=float(np.sum(areas*np.maximum(0,-transformed[:,2]-.35)*(1+3*front)))
            rows.append((cost,tilt,yaw,rotation))
    return sorted(rows,key=lambda x:x[0])


def scan_layers(volume,max_supports,origin,pitch,inverse,protect_front=True):
    points=[];unresolved=[];previous=np.zeros(volume.shape[:2],bool)
    structure=ndi.generate_binary_structure(2,1)
    for z in range(volume.shape[2]):
        current=volume[:,:,z]
        labels,count=ndi.label(current,structure)
        for index in range(1,count+1):
            region=labels==index
            if (region&previous).any():continue
            distance=ndi.distance_transform_edt(region);xy=np.argwhere(region)
            order=np.argsort(distance[region])[::-1]
            chosen=None
            for row in xy[order[:200]]:
                world=np.array([origin[0]+row[0]*pitch,origin[1]+row[1]*pitch,origin[2]+z*pitch])
                original=trimesh.transform_points(world[None],inverse)[0]
                # Front face of source STL points toward -Y; protect its central
                # protruding surface conservatively. This is a heuristic region.
                if protect_front and original[1]<-2.0 and abs(original[0])<5.:continue
                chosen=world;break
            if chosen is None or len(points)>=max_supports:
                unresolved.append({'layer':z,'pixels':int(region.sum()),'reason':'protected_front_region_or_support_budget'})
            else:points.append(chosen.tolist())
        previous=current
    return points,unresolved


def frustum(radius_top,radius_base,height,sections=12):
    angles=np.arange(sections)*2*np.pi/sections
    circles=np.column_stack((np.cos(angles),np.sin(angles)))
    vertices=np.vstack((np.c_[circles*radius_base,np.zeros(sections)],np.c_[circles*radius_top,np.full(sections,height)],[[0,0,0],[0,0,height]]))
    faces=[]
    for i in range(sections):
        j=(i+1)%sections
        faces.extend([[i,j,i+sections],[j,j+sections,i+sections],[2*sections,j,i],[2*sections+1,i+sections,j+sections]])
    mesh=trimesh.Trimesh(vertices,faces,process=False);mesh.fix_normals();return mesh


def prepare_supports(stl,output,settings):
    output.mkdir();mesh=trimesh.load(stl,force='mesh');cost,tilt,yaw,rotation=candidate_orientations(mesh)[0]
    head=mesh.copy();head.apply_transform(rotation)
    shift=np.eye(4);shift[2,3]=settings['lift_mm']-head.bounds[0,2];head.apply_transform(shift)
    transform=shift@rotation;inverse=np.linalg.inv(transform)
    head.export(output/'head_oriented_mm.stl')
    pitch=settings['voxel_pitch_mm']
    if np.prod(np.ceil(head.extents/pitch)+3)>12000000:raise ValueError('Support precheck exceeds voxel budget')
    vox=head.voxelized(pitch).fill()
    volume=vox.matrix;origin=vox.transform[:3,3]
    if volume.size>12000000:raise ValueError('Support precheck exceeds voxel budget')
    anchors,unresolved=scan_layers(volume,settings['max_supports'],origin,pitch,inverse,settings['protect_front'])
    supports=[]
    # Every shaft starts at the raft; accidental model/shaft contacts are checked
    # below and explicitly reported, never hidden behind a support-ready claim.
    radius=settings['shaft_radius_mm'];tip=settings['tip_radius_mm'];tip_length=settings['tip_length_mm']
    for point in anchors:
        x,y,z=point;end=z+pitch
        if end<=.8:continue
        shaft_top=max(.81,end-tip_length)
        shaft=trimesh.creation.cylinder(radius=radius,height=shaft_top-.4,sections=12)
        shaft.apply_translation([x,y,(shaft_top+.4)/2]);supports.append(shaft)
        cone=frustum(radius_top=tip,radius_base=radius,height=end-shaft_top,sections=12)
        cone.apply_translation([x,y,shaft_top]);supports.append(cone)
    center=head.bounds.mean(0);raft=trimesh.creation.cylinder(radius=float(np.linalg.norm(head.extents[:2])/2+1),height=.8,sections=48)
    raft.apply_translation([center[0],center[1],.4]);supports.append(raft)
    assembly=trimesh.util.concatenate([head]+supports);support_mesh=trimesh.util.concatenate(supports)
    support_mesh.export(output/'supports_mm.stl');assembly.export(output/'assembly_unioned_not_verified_mm.stl')
    union_report={'attempted':True,'watertight':False,'components':None}
    try:
        united=trimesh.boolean.union([head]+supports,engine='manifold',check_volume=True)
        united.merge_vertices();parts=united.split(only_watertight=False)
        union_report.update(watertight=bool(united.is_watertight),components=len(parts))
        if united.is_watertight and len(parts)==1:united.export(output/'assembly_union_mm.stl')
    except Exception as exc:union_report['error']=str(exc)
    # Shaft intersections with the original head are possible and must be flagged.
    occupancy=volume.copy();collision_samples=0
    for x,y,z in anchors:
        ij=np.rint((np.array([x,y])-origin[:2])/pitch).astype(int)
        stop=int(np.floor((z-tip_length-origin[2])/pitch))
        if stop>0:collision_samples+=int(volume[ij[0],ij[1],:stop].sum())
    report={'status':'EXPERIMENTAL_SUPPORTS_EXPORTED','orientation_tilt_x_deg':tilt,'orientation_yaw_z_deg':yaw,
        'orientation_score_proxy_mm2':cost,'source_to_oriented_stl':transform.tolist(),'voxel_pitch_mm':pitch,
        'head_voxel_shape':list(volume.shape),'support_anchor_count':len(anchors),'unresolved_islands':unresolved,
        'shaft_model_collision_voxels':collision_samples,'support_contacts':anchors,'protected_front_heuristic':settings['protect_front'],
        'supports_generated':True,'support_strength_verified':False,'support_mesh_union':union_report,'support_mesh_union_verified':bool(union_report['watertight'] and union_report['components']==1),
        'sliced_layers_checked':False,'machine_format_checked':False,'ready_for_machine':False,
        'scope':'coarse voxel connected-component check, not actual printer layers; shafts may touch model; inspection artifacts, not a production support strategy'}
    (output/'support_report.json').write_text(json.dumps(report,indent=2))
    return report
