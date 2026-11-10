"""Local mesh shading and arm-contact experiments; no identity fitting."""
import argparse,json
from pathlib import Path
import numpy as np

def smooth_seam_normals(vertices,faces):
    v=np.asarray(vertices,float);f=np.asarray(faces,int)
    _,inverse=np.unique(np.round(v,7),axis=0,return_inverse=True)
    normal=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    sums=np.zeros((inverse.max()+1,3),float)
    for i in range(3):np.add.at(sums,inverse[f[:,i]],normal)
    sums/=np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-30)
    return sums[inverse]

def arm_weights(vertices):
    v=np.asarray(vertices,float)
    def smooth(x):
        x=np.clip(x,0,1);return x*x*(3-2*x)
    return smooth((v[:,2]-.76)/.12)*smooth((-.060-v[:,0])/.025)

def boundary_loops(vertices,faces):
    """Ordered manifold boundary rings after welding coincident UV seam vertices."""
    _,first,inverse=np.unique(np.round(vertices,7),axis=0,return_index=True,return_inverse=True);unique=np.asarray(vertices)[first];f=inverse[np.asarray(faces)]
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);edges,count=np.unique(edges,axis=0,return_counts=True);edges=edges[count==1]
    adjacency={int(i):[] for i in np.unique(edges)}
    for a,b in edges:adjacency[int(a)].append(int(b));adjacency[int(b)].append(int(a))
    loops=[];seen=set()
    for start in adjacency:
        if start in seen:continue
        component=[];stack=[start]
        while stack:
            item=stack.pop()
            if item in seen:continue
            seen.add(item);component.append(item);stack.extend(adjacency[item])
        if not all(len(adjacency[i])==2 for i in component):continue
        ordered=[];previous=None;current=start
        while current not in ordered:
            ordered.append(current);a,b=adjacency[current];nxt=a if a!=previous else b;previous,current=current,nxt
        loops.append(unique[ordered])
    return loops

def bridge_rings(first,second):
    """Zip two ordered rings without adding boundary samples or changing vertices."""
    a=np.array(first,float,copy=True);b=np.array(second,float,copy=True)
    def orient(v):
        area=np.sum(v[:,0]*np.roll(v[:,1],-1)-np.roll(v[:,0],-1)*v[:,1])
        return v if area>0 else v[::-1]
    a=orient(a);b=orient(b);a=np.roll(a,-int(np.argmin(a[:,0])),axis=0);b=np.roll(b,-int(np.linalg.norm(b-a[0],axis=1).argmin()),axis=0)
    def knots(v):
        length=np.linalg.norm(np.roll(v,-1,axis=0)-v,axis=1);return np.r_[0,np.cumsum(length)/length.sum()]
    ka,kb=knots(a),knots(b);na,nb=len(a),len(b);i=j=0;faces=[]
    while i<na or j<nb:
        if j>=nb or (i<na and ka[i+1]<=kb[j+1]):faces.append([i%na,(i+1)%na,na+j%nb]);i+=1
        else:faces.append([i%na,na+(j+1)%nb,na+j%nb]);j+=1
    return np.vstack([a,b]),np.array(faces)

def scalp_envelope_scale(vertices,center,radii,base,height=.034):
    """Conservative XY expansion of the existing straight-band inner envelope."""
    v=np.asarray(vertices,float);center=np.asarray(center,float);radii=np.asarray(radii,float)-.003
    if (radii<=0).any() or height<=0 or not np.isfinite(v).all():raise ValueError('Invalid cap envelope')
    scalp=v[v[:,2]>=base]
    if not len(scalp):raise ValueError('No scalp samples above band')
    profile=1-.035*(scalp[:,2]-base)/height
    if (profile<=0).any():raise ValueError('Scalp outside supported cap height')
    q=np.linalg.norm((scalp[:,:2]-center)/(profile[:,None]*radii),axis=1)
    factor=max(1.,float(q.max())*1.02)
    return factor,{'scalp_samples':len(scalp),'outside_before':int((q>1).sum()),'outside_after':int((q/factor>1).sum()),'max_inner_envelope_ratio_after':float((q/factor).max()),'scope':'Sampled head vertices against fitted elliptical inner-band profile; not full triangle collision certification'}

def main():
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree
    from .service import sha,write
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];base=root/'runs/assembly-cohesion-01'
    prior=root/'runs/graduation-assembly-04/balanced/assembly.glb';restored=base/'restored-body-v5.glb';a.output=a.output.resolve()
    if a.output.exists():raise ValueError('Use a fresh attempt')
    prior_report=json.loads((root/'runs/graduation-assembly-04/report.json').read_text())
    if sha(prior)!=prior_report['cases']['balanced']['glb_sha256']:raise ValueError('Prior assembly pin changed')
    scene0=trimesh.load(prior,force='scene',process=False);body=next(iter(trimesh.load(restored,force='scene',process=False).geometry.values())).copy();head=scene0.geometry['Personalized_Complete_Head'].copy()
    original_head=head.vertices.copy();head.vertex_normals=smooth_seam_normals(head.vertices,head.faces)
    if not np.array_equal(head.vertices,original_head):raise ValueError('Shading changed head geometry')
    cap_meta=prior_report['cases']['balanced']['cap'];cup=scene0.geometry['Cap_Cup'].copy();cup_scale,cup_audit=scalp_envelope_scale(head.vertices,cap_meta['center_xy'],cap_meta['radii_xy'],cap_meta['band_base_working_z'])
    cup.vertices[:,:2]=np.asarray(cap_meta['center_xy'])+(cup.vertices[:,:2]-cap_meta['center_xy'])*cup_scale
    cup.vertex_normals=smooth_seam_normals(cup.vertices,cup.faces)
    loops=boundary_loops(body.vertices,body.faces);patches=[];hand_patch=None;repairs=[];tree=cKDTree(body.vertices)
    atlas=np.asarray(body.visual.material.baseColorTexture.convert('RGB'))
    for loop in loops:
        hand=loop[:,0].max()<-.075 and loop[:,2].min()>.85
        small=len(loop)<25 and np.ptp(loop,axis=0).max()<.015
        if not ((hand and np.ptp(loop,axis=0).max()<.12) or small):continue
        center=loop.mean(0);contact_point=center.copy();rounded=hand and np.ptp(loop[:,0])<1e-5
        if rounded:
            depth=.014 if center[2]<.948 else .004;contact_point[0]+=depth;n=len(loop);rings=5;rows=[];faces=[]
            for row in range(rings):
                angle=(np.pi/2)*row/rings;points=center+(loop-center)*np.cos(angle);points[:,0]=center[0]+depth*np.sin(angle);rows.extend(points)
            rows.append(contact_point);v=np.array(rows)
            for row in range(rings-1):
                for i in range(n):
                    a0=row*n+i;b0=row*n+(i+1)%n;c0=a0+n;d0=b0+n;faces.extend([[a0,b0,c0],[b0,d0,c0]])
            for i in range(n):faces.append([(rings-1)*n+i,(rings-1)*n+(i+1)%n,rings*n])
            patch=trimesh.Trimesh(v,faces,process=False)
            if patch.face_normals[:,0].mean()<0:patch.invert()
        else:
            v=np.vstack([loop,center]);f=np.array([[i,(i+1)%len(loop),len(loop)] for i in range(len(loop))]);patch=trimesh.Trimesh(v,f,process=False)
        _,nearest=tree.query(loop);uv=body.visual.uv[nearest];xy=np.rint(uv*[atlas.shape[1]-1,atlas.shape[0]-1]).astype(int);pixels=atlas[(atlas.shape[0]-1-xy[:,1]).clip(0,atlas.shape[0]-1),xy[:,0].clip(0,atlas.shape[1]-1)]
        if hand:
            rgb=pixels.astype(float)/255;hi=rgb.max(1);lo=rgb.min(1);sat=(hi-lo)/np.maximum(hi,1e-9);hue=(rgb[:,1]-rgb[:,2])/np.maximum(6*(hi-lo),1e-9)
            skin=(rgb[:,0]>=rgb[:,1])&(rgb[:,1]>=rgb[:,2])&(hue<.105)&(sat>.07)&(sat<.7)&(hi>.35)
            if skin.sum()>=3:pixels=pixels[skin]
        color=tuple(np.rint(np.median(pixels,axis=0)).astype(np.uint8))
        material=trimesh.visual.material.PBRMaterial(name='Inferred_Contact_Fill',baseColorTexture=Image.new('RGB',(2,2),color),roughnessFactor=1,metallicFactor=0)
        patch.visual=trimesh.visual.TextureVisuals(uv=np.full((len(v),2),.5),material=material);patches.append(patch)
        if hand and (hand_patch is None or abs(contact_point[2]-.956)<abs(hand_patch[1][2]-.956)):hand_patch=(loop,contact_point)
        repairs.append({'boundary_vertices':len(loop),'bounds':[loop.min(0).tolist(),loop.max(0).tolist()],'hand_region':bool(hand),'rounded_fingertip':bool(rounded),'surface_and_color_inferred':True})
    if hand_patch is None:raise ValueError('No supported fingertip contact patch found')
    c=body.triangles_center;region=(c[:,0]<-.08)&(c[:,0]>-.18)&(c[:,2]>.88)&(c[:,2]<.94)&(c[:,1]>-.10)&(c[:,1]<.12);ids=np.flatnonzero(region)
    uv=body.visual.uv[body.faces[ids]].mean(1);xy=np.rint(uv*[atlas.shape[1]-1,atlas.shape[0]-1]).astype(int);rgb=atlas[(atlas.shape[0]-1-xy[:,1]).clip(0,atlas.shape[0]-1),xy[:,0].clip(0,atlas.shape[1]-1)].astype(float)/255;hi=rgb.max(1);lo=rgb.min(1);sat=(hi-lo)/np.maximum(hi,1e-9);hue=(rgb[:,1]-rgb[:,2])/np.maximum(6*(hi-lo),1e-9);gold=(hue>.105)&(hue<.21)&(sat>.05)&(rgb[:,0]>=rgb[:,1])&(rgb[:,1]>=rgb[:,2])
    if not gold.any():raise ValueError('Cannot find the retained gold tassel contact marker')
    gold_ids=ids[gold];legacy_tassel=c[gold_ids[np.argmax(c[gold_ids,2])]]
    neck_body=[q for q in loops if q[:,2].min()>.76 and q[:,2].max()<.85 and q[:,0].min()>-.10 and q[:,0].max()<.10]
    neck_head=[q for q in boundary_loops(head.vertices,head.faces) if np.abs(q[:,2]-.765).max()<1e-5]
    if len(neck_body)!=1 or len(neck_head)!=1:raise ValueError('Need unambiguous neck boundary rings')
    nv,nf=bridge_rings(neck_body[0],neck_head[0]);neck_bridge=trimesh.Trimesh(nv,nf,process=False)
    _,nearest=cKDTree(head.vertices).query(neck_head[0]);uv=head.visual.uv[nearest];image=np.asarray(head.visual.material.baseColorTexture.convert('RGB'));xy=np.rint(uv*[image.shape[1]-1,image.shape[0]-1]).astype(int);pixels=image[(image.shape[0]-1-xy[:,1]).clip(0,image.shape[0]-1),xy[:,0].clip(0,image.shape[1]-1)];color=tuple(np.rint(np.median(pixels,axis=0)).astype(np.uint8))
    neck_bridge.visual=trimesh.visual.TextureVisuals(uv=np.full((len(nv),2),.5),material=trimesh.visual.material.PBRMaterial(name='Inferred_Neck_Transition',baseColorTexture=Image.new('RGB',(2,2),color),roughnessFactor=1,metallicFactor=0))
    radial=neck_bridge.triangles_center.copy();radial[:,:2]-=nv[:,:2].mean(0);radial[:,2]=0
    if np.einsum('ij,ij->i',neck_bridge.face_normals,radial).mean()<0:neck_bridge.invert()
    source_contact=hand_patch[1];board=scene0.geometry['Cap_Board'];target=np.array([board.bounds[0,0]-.0003,np.clip(source_contact[1],board.bounds[0,1]+.01,board.bounds[1,1]-.01),board.bounds[:,2].mean()]);delta=target-source_contact
    # Enforce a separating X plane for hand triangles whose Y/Z bounds cross the board.
    hand_meshes=[body,*patches];clearance=.0007
    for _ in range(8):
        max_x=-np.inf
        for mesh in hand_meshes:
            moved=mesh.vertices+arm_weights(mesh.vertices)[:,None]*delta;tri=moved[mesh.faces]
            select=(tri[:,:,0].max(1)<-.065)&(tri[:,:,2].min(1)>.90)&(tri[:,:,1].max(1)>=board.bounds[0,1])&(tri[:,:,1].min(1)<=board.bounds[1,1])&(tri[:,:,2].max(1)>=board.bounds[0,2])&(tri[:,:,2].min(1)<=board.bounds[1,2])
            if select.any():max_x=max(max_x,float(tri[select,:,0].max()))
        correction=max_x-(board.bounds[0,0]-clearance)
        if correction<=1e-8:break
        delta[0]-=correction
    else:raise ValueError('Could not clear the board contact region')
    if np.linalg.norm(delta)>.04:raise ValueError('Contact correction exceeds arm budget')
    a.output.mkdir(parents=True);axes=np.eye(4);axes[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
    report={'policy':'scalp_clearance_hand_neck_cohesion_v7','input_sha256':{str(q):sha(q) for q in [prior,restored,base/'resegment-report-v5.json',base/'restored-selection-v5.npz']},'code_sha256':sha(Path(__file__)),'cap_xy_expansion':cup_scale,'cap_envelope_audit':cup_audit,'hand_repairs':repairs,'source_contact_working':source_contact.tolist(),'requested_contact_working':target.tolist(),'correction_working':delta.tolist(),'hand_board_separating_x_clearance':float(board.bounds[0,0]-max_x),'legacy_tassel_marker_working':legacy_tassel.tolist(),'duplicate_tassel_added':False,'head_vertices_unchanged':True,'head_shading':'area-weighted normals shared across coincident UV seam vertices; geometry is not subdivided','hand_cut':'colour-weighted geodesic selection and material delamination; contact surfaces inferred','neck_bridge':{'body_ring_vertices':len(neck_body[0]),'head_ring_vertices':len(neck_head[0]),'triangles':len(nf),'skin_inferred':True,'intersection_checked':False},'neck_join_validated':False,'likeness_verified':False,'ready_for_machine':False,'cases':{}}
    for name,move in [('restored',False),('contact',True)]:
        folder=a.output/name;folder.mkdir();b=body.copy();w=arm_weights(b.vertices)
        old=b.vertices.copy()
        if move:b.vertices=old+w[:,None]*delta
        normal0=np.cross(old[b.faces[:,1]]-old[b.faces[:,0]],old[b.faces[:,2]]-old[b.faces[:,0]]);normal1=np.cross(b.vertices[b.faces[:,1]]-b.vertices[b.faces[:,0]],b.vertices[b.faces[:,2]]-b.vertices[b.faces[:,0]])
        valid=np.linalg.norm(normal0,axis=1)>1e-14;reversed_count=int((np.einsum('ij,ij->i',normal0[valid],normal1[valid])<=0).sum())
        if reversed_count:raise ValueError('Arm correction reversed triangles')
        b.vertex_normals=smooth_seam_normals(b.vertices,b.faces);scene=trimesh.Scene();scene.add_geometry(b,node_name='Body_Restored',geom_name='Body_Restored');scene.add_geometry(head,node_name='Personalized_Complete_Head',geom_name='Personalized_Complete_Head')
        bridge=neck_bridge.copy();combined_v=np.vstack([b.vertices,head.vertices,bridge.vertices]);combined_f=np.vstack([b.faces,head.faces+len(b.vertices),bridge.faces+len(b.vertices)+len(head.vertices)]);normals=smooth_seam_normals(combined_v,combined_f)
        b.vertex_normals=normals[:len(b.vertices)];head.vertex_normals=normals[len(b.vertices):len(b.vertices)+len(head.vertices)];bridge.vertex_normals=normals[-len(bridge.vertices):]
        scene.add_geometry(bridge,node_name='Neck_Transition',geom_name='Neck_Transition')
        scene.add_geometry(cup.copy(),node_name='Cap_Cup',geom_name='Cap_Cup');scene.add_geometry(board.copy(),node_name='Cap_Board',geom_name='Cap_Board')
        for i,patch in enumerate(patches):
            patch=patch.copy()
            if move:patch.vertices+=arm_weights(patch.vertices)[:,None]*delta
            scene.add_geometry(patch,node_name='Contact_Fill_'+str(i),geom_name='Contact_Fill_'+str(i))
        contact=source_contact.copy()
        if move:contact+=arm_weights(source_contact[None,:])[0]*delta
        top=board.bounds[1,2]+.0015;center=board.bounds.mean(0)
        tassel_contact=legacy_tassel+(arm_weights(legacy_tassel[None,:])[0]*delta if move else 0)
        route=[np.array([center[0],center[1],top]),np.array([target[0],target[1],top]),contact,tassel_contact]
        cords=[]
        for start,end in zip(route[:-1],route[1:]):
            if np.linalg.norm(end-start)<1e-6:continue
            piece=trimesh.creation.cylinder(radius=.0012,height=np.linalg.norm(end-start),sections=12);mat=trimesh.geometry.align_vectors([0,0,1],end-start);mat[:3,3]=(start+end)/2;piece.apply_transform(mat);cords.append(piece)
        cord=trimesh.util.concatenate(cords);cord.visual=trimesh.visual.TextureVisuals(material=scene0.geometry['Cap_Cord'].visual.material)
        scene.add_geometry(cord,node_name='Cap_Cord',geom_name='Cap_Cord')
        scene.apply_transform(np.linalg.inv(axes));scene.export(folder/'assembly.glb',include_normals=True)
        reread=trimesh.load(folder/'assembly.glb',force='scene',process=False);new=reread.geometry['Personalized_Complete_Head']
        if not np.array_equal(new.vertices,original_head):raise ValueError('Head geometry changed after export')
        report['cases'][name]={'glb_sha256':sha(folder/'assembly.glb'),'arm_moved':move,'arm_max_displacement_working':float(np.linalg.norm(b.vertices-old,axis=1).max()),'normal_reversals':reversed_count,'actual_marker_working':contact.tolist(),'marker_target_distance':float(np.linalg.norm(contact-target)),'untouched_vertex_count':int((w==0).sum()),'head_positions_equal':True,'fingertip_fill':'inferred fan cap; not a scan of hidden skin','full_collision_check':False,'ready_for_machine':False}
    write(a.output/'report.json',report);print('COHESION_CANDIDATES_EXPORTED')

if __name__=='__main__':main()
