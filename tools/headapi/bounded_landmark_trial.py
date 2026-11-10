"""CPU normal-displacement trial: front eye/lip data, frozen cameras and small bounds."""
import json
import sys
from uuid import uuid4
from pathlib import Path
import numpy as np
import trimesh
from scipy import sparse
from scipy.optimize import lsq_linear
from PIL import Image, ImageDraw
from tools.headapi.camera_diagnostics import sha, visible
from tools.headapi.face_export import export_face

ROOT=Path(__file__).resolve().parents[2]


def project(points,P):
    h=np.column_stack((points,np.ones(len(points))))@P.T
    return h[:,:2]/h[:,2,None]


def run(guard=False):
    output=ROOT/'runs/experiments'/('rdf-bounded-eyes-mouth-'+uuid4().hex);output.mkdir(parents=True,exist_ok=False)
    print(output,flush=True)
    source=ROOT/'runs/headapi/9bf00af2-3a4b-40e1-bd08-5706c66ef19d'
    fit=ROOT/'runs/experiments/rdf-ab-696a22fbe38149bf96d9d6b89ec65f5e/B_three'
    bundle=fit/'projection_bundle.npz'
    if sha(bundle)!=json.loads((fit/'report.json').read_text())['projection_bundle_sha256']:raise ValueError('Bundle changed')
    asset=ROOT/'runs/experiments/rdf-mediapipe-105-01/mediapipe_landmark_embedding.npz'
    if sha(asset)!='8863363013fc6fe3752d4318ea02a1784970200616a54b785920cdd0568816fc':raise ValueError('Embedding changed')
    with np.load(bundle,allow_pickle=False) as b:data={k:b[k].copy() for k in b.files}
    with np.load(asset,allow_pickle=False) as b:fi=b['lmk_face_idx'];weights=b['lmk_b_coords'];ids=b['landmark_indices']
    metadata=json.loads((source/'rdf_crops/report.json').read_text());slots=['front','left','right']
    vertices=data['vertices'].astype(np.float64);faces=data['faces'];P=data['cameras'][0]
    mesh=trimesh.Trimesh(vertices,faces,process=False);normals=np.asarray(mesh.vertex_normals).copy()
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
    height=float(np.ptp(vertices[:,1]));observed=np.asarray(metadata['views']['front']['landmarks_crop'])
    eye_distance=float(np.linalg.norm((observed[33]+observed[133]-observed[362]-observed[263])/2))
    previous=json.loads((ROOT/'runs/experiments/rdf-mediapipe-105-01/comparison.json').read_text())['variants']['baseline']
    train=[j for j,p in enumerate(previous['front']['points']) if p['region'] in ('eyes','mouth') and p['mesh_visible'] and p['in_frame']]
    region=set(faces[fi[train]].ravel().tolist());edges=np.unique(mesh.edges_sorted,axis=0)
    for _ in range(2):
        region.update(edges[np.isin(edges[:,0],list(region))|np.isin(edges[:,1],list(region))].ravel().tolist())
    # Protect all direct nose/brow embedding supports; eye balls are separate components.
    protected=set(faces[fi[np.r_[0:20,52:65]]].ravel().tolist())
    graph=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices)))
    _,labels=sparse.csgraph.connected_components(graph,directed=False)
    head_label=int(np.argmax(np.bincount(labels)));active=np.array(sorted(v for v in region-protected if labels[v]==head_label))
    lookup={int(v):i for i,v in enumerate(active)}
    matrix=np.zeros((2*len(train),len(active)));target=[]
    for row,j in enumerate(train):
        vv=faces[fi[j]];point=(vertices[vv]*weights[j,:,None]).sum(0);h=P@np.r_[point,1]
        xy=h[:2]/h[2];target.extend((observed[ids[j]]-xy)/eye_distance)
        jac=(P[:2,:3]*h[2]-h[:2,None]*P[2,:3])/h[2]**2
        for v,w in zip(vv,weights[j]):
            if int(v) in lookup:matrix[2*row:2*row+2,lookup[int(v)]]=jac@(normals[v]*height*w)/eye_distance
    rr=[];cc=[];values=[];edge_row=0
    for a,b in edges:
        if int(a) not in lookup and int(b) not in lookup:continue
        for v,sign in [(a,1),(b,-1)]:
            if int(v) in lookup:rr.append(edge_row);cc.append(lookup[int(v)]);values.append(sign)
        edge_row+=1
    smooth=sparse.coo_matrix((values,(rr,cc)),shape=(edge_row,len(active))).tocsr()
    A=sparse.vstack([sparse.csr_matrix(matrix),smooth,sparse.eye(len(active))*0.05],format='csr')
    rhs=np.r_[target,np.zeros(edge_row+len(active))]
    result={'schema_version':1,'algorithm':'one linearized reprojection step; bounded normal offsets; edge smoothing=1, anchor=0.05',
        'guard_policy':'posthoc backtracking: face normal change <=30deg, area ratio >=0.2, edge length change <=15%' if guard else 'none; diagnostic candidates only',
        'train_views':['front'],'evaluation_views':['left','right'],
        'held_out_scope':'profiles held out of refinement only; baseline used all three views',
        'active_vertices':len(active),'training_landmarks':len(train),'camera_changed':False,
        'weights_changed':False,'semantic_mapping_verified':False,'self_intersections_verified':False,
        'input_sha256':{str(p):sha(p) for p in [bundle,asset,source/'rdf_crops/report.json',Path(__file__)]},'variants':{}}
    base_cross=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    for fraction in [0.001,0.003,0.006]:
        name='bound_'+str(fraction).replace('.','_');folder=output/name;folder.mkdir()
        solution=lsq_linear(A,rhs,bounds=(-fraction,fraction),lsmr_tol='auto',max_iter=150)
        if not solution.success:raise RuntimeError('Bounded solve did not converge: '+solution.message)
        delta=np.zeros_like(vertices);delta[active]=normals[active]*height*solution.x[:,None]
        if np.max(np.linalg.norm(delta,axis=1))>height*fraction*(1+1e-8):raise ValueError('Displacement bound exceeded')
        alpha=1.0
        if guard:
            area0=np.linalg.norm(base_cross,axis=1)
            edge0=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
            for _ in range(20):
                test=vertices+alpha*delta
                cross=np.cross(test[faces[:,1]]-test[faces[:,0]],test[faces[:,2]]-test[faces[:,0]])
                cos=np.einsum('ij,ij->i',base_cross,cross)/np.maximum(area0*np.linalg.norm(cross,axis=1),1e-20)
                edge1=np.linalg.norm(test[edges[:,0]]-test[edges[:,1]],axis=1)
                if np.all(cos>=np.cos(np.radians(30))) and np.all(np.linalg.norm(cross,axis=1)>=.2*area0) and np.max(np.abs(edge1/np.maximum(edge0,1e-15)-1))<=.15:break
                alpha*=.5
            else:raise ValueError('No bounded deformation satisfies trial geometry guards')
            delta*=alpha
        candidate=vertices+delta;new_cross=np.cross(candidate[faces[:,1]]-candidate[faces[:,0]],candidate[faces[:,2]]-candidate[faces[:,0]])
        cosine=np.einsum('ij,ij->i',base_cross,new_cross)/np.maximum(np.linalg.norm(base_cross,axis=1)*np.linalg.norm(new_cross,axis=1),1e-20)
        new_data={**data,'vertices':candidate,'view_vertices':data['view_vertices']+delta[None]}
        np.savez_compressed(folder/'projection_bundle.npz',**new_data)
        export_face(folder/'projection_bundle.npz',source/'rdf_crops',slots,folder)
        row={'fraction_head_height':fraction,'guard_scale':alpha,'max_displacement_native':float(np.linalg.norm(delta,axis=1).max()),
            'changed_vertices':int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-10)),
            'triangle_normal_reversals':int((cosine<=0).sum()),
            'degenerate_triangles':int((np.linalg.norm(new_cross,axis=1)<1e-14).sum()),
            'max_triangle_normal_change_degrees':float(np.degrees(np.arccos(np.clip(cosine,-1,1))).max()),
            'bundle_sha256':sha(folder/'projection_bundle.npz'),'views':{}}
        result['variants'][name]=row
        for i,view in enumerate(slots):
            V=new_data['view_vertices'][i];camera=data['cameras'][i]
            points=(V[faces[fi]]*weights[:,:,None]).sum(1);xy=project(points,camera)
            obs=np.asarray(metadata['views'][view]['landmarks_crop']);image=Image.open(source/'rdf_crops'/(view+'.png')).convert('RGB');draw=ImageDraw.Draw(image)
            import cv2
            c=cv2.decomposeProjectionMatrix(camera)[2];c=(c[:3]/c[3]).reshape(3)
            records=[]
            for j,base in enumerate(previous[view]['points']):
                # Freeze baseline visibility for fair evaluation; also report losses.
                if not base['mesh_visible'] or not base['in_frame']:continue
                err=float(np.linalg.norm(xy[j]-obs[ids[j]]));vis=visible(V,faces,points[j],c)
                records.append({'mp':int(ids[j]),'region':base['region'],'error_px':err,'baseline_error_px':base['residual_px'],'candidate_visible':vis})
                if base['region'] in ('eyes','mouth'):
                    draw.line([tuple(xy[j]),tuple(obs[ids[j]])],fill='yellow',width=1)
            row['views'][view]={'regions':{r:{'baseline_mean_px':float(np.mean([p['baseline_error_px'] for p in records if p['region']==r])),
                'candidate_mean_px':float(np.mean([p['error_px'] for p in records if p['region']==r])),
                'visibility_lost':sum(not p['candidate_visible'] for p in records if p['region']==r)}
                for r in ('eyes','mouth','nose','brows') if any(p['region']==r for p in records)},'points':records}
            image.save(folder/(view+'.overlay.png'))
        (output/'experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({n:{'reversals':r['triangle_normal_reversals'],'views':{v:d['regions'] for v,d in r['views'].items()}} for n,r in result['variants'].items()},indent=2))


if __name__=='__main__':run(guard='--guard' in sys.argv)
