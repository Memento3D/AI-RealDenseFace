"""Offline proportion/pose trial, retaining a complete head and separate cap."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import trimesh
from .service import sha,write
from .head_assembly_trial import condition_neck

def upright_transform(reference,source_eye,target_eye,scale_factor=1):
    reference=np.asarray(reference,float);source_eye=np.asarray(source_eye,float);target_eye=np.asarray(target_eye,float)
    if reference.shape!=(4,4) or not np.isfinite(reference).all() or not 0.8<=scale_factor<=1.1:raise ValueError('Invalid pose inputs')
    scale=float(np.cbrt(np.linalg.det(reference[:3,:3])))*scale_factor
    if scale<=0:raise ValueError('Reflection or invalid scale')
    yaw=np.arctan2(reference[1,0],reference[0,0]);c,s=np.cos(yaw),np.sin(yaw)
    matrix=np.eye(4);matrix[:3,:3]=scale*np.array([[c,-s,0],[s,c,0],[0,0,1]])
    matrix[:3,3]=target_eye-matrix[:3,:3]@source_eye
    return matrix

def cap_parts(head):
    top=float(head.vertices[:,2].max());base=top-.028
    scalp=head.vertices[head.vertices[:,2]>base];center=(scalp[:,:2].min(0)+scalp[:,:2].max(0))/2
    radii=(scalp[:,:2].max(0)-scalp[:,:2].min(0))/2+.004
    segments=64;rows=10;verts=[]
    for inner in (False,True):
        for j in range(rows):
            t=j/(rows-1);r=1-.035*t;z=base+t*(.034 if not inner else .031)
            for i in range(segments):
                angle=2*np.pi*i/segments;radius=radii-(.003 if inner else 0)
                verts.append([center[0]+radius[0]*r*np.cos(angle),center[1]+radius[1]*r*np.sin(angle),z])
    faces=[];offset=segments*rows
    for layer in (0,1):
        for j in range(rows-1):
            for i in range(segments):
                a=layer*offset+j*segments+i;b=layer*offset+j*segments+(i+1)%segments;c=a+segments;d=b+segments
                faces.extend([[a,b,c],[b,d,c]] if layer==0 else [[a,c,b],[b,c,d]])
    for row in (0,rows-1):
        for i in range(segments):
            a=row*segments+i;b=row*segments+(i+1)%segments;c=a+offset;d=b+offset
            faces.extend([[a,c,b],[b,c,d]] if row==0 else [[a,b,c],[b,d,c]])
    material=trimesh.visual.material.PBRMaterial(name='Graduation_Cap_Black',baseColorFactor=[0,0,0,255],roughnessFactor=.95,metallicFactor=0)
    cup=trimesh.Trimesh(verts,faces,process=False);cup.visual=trimesh.visual.TextureVisuals(material=material)
    board_width=max(.16,2.35*float(radii.max()))
    board=trimesh.creation.box(extents=[board_width,board_width,.004]);board.apply_translation([*center,top+.007]);board.visual=trimesh.visual.TextureVisuals(material=material)
    gold=trimesh.visual.material.PBRMaterial(name='Cap_Cord_Gold',baseColorFactor=[90,55,10,255],roughnessFactor=.65,metallicFactor=.2)
    points=[np.array([*center,top+.010]),np.array([center[0]-board_width/2,center[1]-.055,top+.010]),np.array([-.140,-.025,.985]),np.array([-.140,-.025,.940])]
    cords=[]
    for a,b in zip(points[:-1],points[1:]):
        direction=b-a;piece=trimesh.creation.cylinder(radius=.0015,height=np.linalg.norm(direction),sections=12)
        matrix=trimesh.geometry.align_vectors([0,0,1],direction);matrix[:3,3]=(a+b)/2;piece.apply_transform(matrix);cords.append(piece)
    cord=trimesh.util.concatenate(cords);cord.visual=trimesh.visual.TextureVisuals(material=gold)
    return {'Cap_Cup':cup,'Cap_Board':board,'Cap_Cord':cord},{'shape_policy':'low_straight_fabric_band_v1','board_width':board_width,'top_working_z':top,'band_base_working_z':base,'center_xy':center.tolist(),'radii_xy':radii.tolist(),'fit':'scalp bounding envelope; no collision/shell-clearance certification'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mouth-variant',type=Path);p.add_argument('--proportion-policy',choices=['upright_v1','balanced_v2'],default='balanced_v2');a=p.parse_args();root=Path(__file__).resolve().parents[2];a.output=a.output.resolve()
    if a.output.exists():raise ValueError('Use a fresh attempt')
    prior=root/'runs/head-assembly-01/attempt-04/neck-low/assembly.glb';prior_report=root/'runs/head-assembly-01/attempt-04/report.json';package=root/'runs/experiments/head-review-package-01';head_path=package/'input04/head.glb';bundle=root/'runs/experiments/api-demo-uv-01/case1/rdf/projection_bundle.npz';embedding=root/'runs/prod-win-02/input04-production-02/color/embedding.npz';mask_path=root/'vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl';target=root/'runs/prod-win-01/src/assets/chibi_templates/graduation_men_phuoc_v1/target_landmarks.json'
    old=json.loads(prior_report.read_text());manifest=json.loads((package/'manifest.json').read_text())
    if sha(prior)!=old['cases']['neck-low']['assembled_sha256'] or sha(head_path)!=manifest['cases']['input04']['sha256']['head.glb']:raise ValueError('Source pin changed')
    if sha(mask_path)!='ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492':raise ValueError('Mask pin changed')
    with np.load(bundle) as b:raw_v=b['vertices'];raw_f=b['faces']
    with np.load(embedding) as e:landmarks=np.einsum('ij,ijk->ik',e['full_weights'],raw_v[raw_f[e['full_faces']]])
    axes=np.eye(4);axes[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]];landmarks=landmarks@axes[:3,:3].T
    annotations=json.loads(target.read_text());eye_ids=[36,39,42,45];target_eye=np.array([annotations['points'][str(i)]['point'] for i in eye_ids]).mean(0)
    with mask_path.open('rb') as stream:neck_ids=pickle.load(stream,encoding='latin1')['neck']
    source_head=trimesh.load(head_path,force='mesh',process=False)
    if not np.array_equal(source_head.vertices.astype(np.float32),raw_v[raw_f].reshape(-1,3).astype(np.float32)):raise ValueError('Semantic topology binding differs')
    neck_mask=np.isin(raw_f.ravel(),neck_ids);prior_scene=trimesh.load(prior,force='scene',process=False);body=prior_scene.geometry['Body_Template'].copy();a.output.mkdir(parents=True)
    variant=None
    if a.mouth_variant:
        with np.load(a.mouth_variant) as b:variant=b['vertices'].copy();variant_faces=b['faces'];variant_identity=b['identity']
        with np.load(bundle) as b:
            if not np.array_equal(variant_faces,raw_f) or not np.array_equal(variant_identity,b['identity']):raise ValueError('Expression trial changed topology or identity coefficients')
        if not np.isfinite(variant).all() or np.linalg.norm(variant-raw_v,axis=1).max()>.01*np.ptp(raw_v[:,1]):raise ValueError('Expression trial exceeded displacement guard')
    report={'policy':'balanced_head_straight_cap_v4','proportion_policy':a.proportion_policy,'input_sha256':{str(q):sha(q) for q in [prior,prior_report,head_path,bundle,embedding,mask_path,target]+([a.mouth_variant] if a.mouth_variant else [])},'code_sha256':sha(Path(__file__)),'fresh_reconstruction':False,'likeness_verified':False,'api_changed':False,'ready_for_machine':False,'cases':{}}
    cases=[('upright',1.,0.,False),('compact',.94,.005,False)]+([('compact-jaw-bounded',.94,.005,True)] if variant is not None else [])
    if a.proportion_policy=='balanced_v2':cases=[('balanced',.90,.013,variant is not None),('slim',.86,.018,variant is not None)]
    for name,factor,lower,use_variant in cases:
        folder=a.output/name;folder.mkdir();anchor=target_eye.copy();anchor[2]-=lower
        matrix=upright_transform(old['alignment_matrix_working'],landmarks[eye_ids].mean(0),anchor,factor)
        head=source_head.copy()
        if use_variant:head.vertices=variant[raw_f].reshape(-1,3)
        head.apply_transform(matrix@axes);before=head.vertices.copy();head.vertices=condition_neck(head.vertices,neck_mask)
        if not np.array_equal(before[~neck_mask],head.vertices[~neck_mask]):raise ValueError('Face changed during neck conditioning')
        v,f,uv=trimesh.intersections.slice_faces_plane(head.vertices,head.faces,[0,0,1],[0,0,.765],uv=head.visual.uv)
        head=trimesh.Trimesh(v,f,visual=trimesh.visual.TextureVisuals(uv=uv,material=head.visual.material),process=False)
        scene=trimesh.Scene();scene.add_geometry(body,node_name='Body_Template',geom_name='Body_Template');scene.add_geometry(head,node_name='Personalized_Complete_Head',geom_name='Personalized_Complete_Head')
        parts,fit=cap_parts(head)
        for role,mesh in parts.items():scene.add_geometry(mesh,node_name=role,geom_name=role)
        scene.apply_transform(np.linalg.inv(axes));scene.export(folder/'assembly.glb')
        reread=trimesh.load(folder/'assembly.glb',force='scene',process=False)
        if len(reread.geometry)!=5:raise ValueError('Missing assembly components')
        exported=reread.geometry['Personalized_Complete_Head']
        if not np.allclose(exported.vertices[exported.faces],head.vertices[head.faces],atol=1e-7,rtol=0):raise ValueError('Head export changed')
        report['cases'][name]={'matrix_working':matrix.tolist(),'scale_factor_relative_to_old':factor,'eye_anchor_working':anchor.tolist(),'pitch_roll_removed':True,'non_neck_shape_preserved_up_to_similarity':not use_variant,'identity_coefficients_changed':False,'expression_pose_variant':'bounded jaw axis-angle; see source experiment report' if use_variant else 'original','atlas_changed':False,'cap':fit,'glb_sha256':sha(folder/'assembly.glb'),'components':list(reread.geometry),'neck_join':'overlap preview, not welded','ready_for_machine':False}
    write(a.output/'report.json',report);print('GRADUATION_CANDIDATES_EXPORTED_NEEDS_REVIEW')

if __name__=='__main__':main()
