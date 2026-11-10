"""Per-job skin completion and continuous ear-anchored designed hairline.

Generalizes the buzz-framework contour experiment. No cached person's atlas,
quality map, geometry or skin is used. Hidden skin is explicitly estimated.
"""
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from .demo_appearance import atlas_positions
from .texture_completion_trial import export_texture
from .service import sha, write


def run(job, output, assets):
    from scripts.eye_components import identify
    job, output = Path(job), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle=job/'rdf/projection_bundle.npz'
    with np.load(bundle,allow_pickle=False) as data:
        v,f=data['vertices'],data['faces']
    with np.load(assets['uv_assets'],allow_pickle=False) as data:
        if not np.array_equal(f,data['faces']):raise ValueError('Hairline UV topology mismatch')
        uv=data['uvs'][data['uv_faces']]
    with np.load(job/'rdf_uv/raw/quality_maps.npz',allow_pickle=False) as data:
        index=data['face_index'];skin=data['skin'];confidence=data['confidence']
    atlas=np.asarray(Image.open(job/'rdf_uv/final/face_atlas.png').convert('RGB')).copy()
    original=atlas.copy()
    observed=np.asarray(Image.open(job/'rdf_uv/raw/coverage.png'))>0
    positions=atlas_positions(index,v,f,uv)
    valid=np.isfinite(positions).all(-1)
    labels,anatomy=identify(v,f)
    eye=np.mean(anatomy['eye_centers_native'],axis=0);height=float(np.ptp(v[:,1]))
    reference=observed&valid&(skin>0)&(confidence>.35)
    reference&=(positions[...,1]>eye[1]-.26*height)&(positions[...,1]<eye[1]+.04*height)
    reference&=(np.abs(positions[...,0]-eye[0])<.23*height)
    reference&=(original.mean(-1)>25)&(original.mean(-1)<235)
    ry,rx=np.where(reference)
    if len(ry)<32:raise ValueError('Not enough observed face skin for assembly appearance')
    step=max(1,len(ry)//12000);ry,rx=ry[::step],rx[::step]
    colors=original[ry,rx].astype(float);points=positions[ry,rx]
    median=np.median(colors,axis=0);tree=cKDTree(points)
    estimated=valid&~observed
    fy,fx=np.where(estimated)
    for start in range(0,len(fy),32768):
        y,x=fy[start:start+32768],fx[start:start+32768]
        distance,neighbors=tree.query(positions[y,x],k=min(8,len(points)))
        weight=1/np.maximum(distance,.001*height)**2
        weight/=weight.sum(1,keepdims=True)
        local=(colors[neighbors]*weight[...,None]).sum(1)
        atlas[y,x]=np.clip(.4*local+.6*median,0,255).astype(np.uint8)
    with assets['semantic_masks'].open('rb') as stream:
        masks=pickle.load(stream,encoding='latin1')
    ear=np.zeros(len(v),bool)
    for name in ('left_ear','right_ear'):ear[masks[name]]=True
    yy,xx=np.where(index>=0);fi=index[yy,xx]
    pos=positions[yy,xx]
    hairalpha=np.zeros(len(pos));anchors=[]
    for sign,name in [(1,'left_ear'),(-1,'right_ear')]:
        ep=v[masks[name]];front=float(ep[:,2].max());back=float(ep[:,2].min())
        bottom,top=float(ep[:,1].min()),float(ep[:,1].max())
        zn=np.array([v[:,2].min()-.02*height,back-.025*height,
            front-.012*height,front+.018*height,front+.070*height,v[:,2].max()+.02*height])
        yn=np.array([eye[1]-.34*height,eye[1]-.34*height,
            bottom+.28*(top-bottom),bottom+.28*(top-bottom),eye[1]+.17*height,eye[1]+.17*height])
        if not (np.diff(zn)>0).all():raise ValueError('Hairline anchors are not ordered')
        use=pos[:,0]*sign>=0
        boundary=PchipInterpolator(zn,yn)(pos[use,2])
        hairalpha[use]=np.clip(.5+(pos[use,1]-boundary)/(.012*height),0,1)
        anchors.append({'ear':name,'depth_nodes':zn.tolist(),'height_nodes':yn.tolist()})
    hairalpha*=labels[fi]==0
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    length=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    graph=coo_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v))).tocsr()
    outer=dijkstra(graph,directed=False,indices=np.where(ear)[0],min_only=True)
    inner=dijkstra(graph,directed=False,indices=np.where(~ear)[0],min_only=True)
    signed=np.where(ear,-inner,outer);signed=np.where(np.isfinite(signed),signed,10)
    # Reuse barycentric ownership calculated by atlas_positions, via a scalar
    # encoded into all three coordinates (linear interpolation is identical).
    scalar=atlas_positions(index,np.repeat(signed[:,None],3,axis=1),f,uv)[yy,xx,0]
    alpha=hairalpha*np.clip((scalar-.002*height)/(.001*height),0,1)
    base=atlas[yy,xx].copy()
    atlas[yy,xx]=np.clip(base*(1-alpha[:,None])+np.array([50,46,43])*alpha[:,None],0,255).astype(np.uint8)
    changed_hair=np.zeros(valid.shape,bool);changed_hair[yy,xx]=alpha>0
    protected=observed&~changed_hair
    if not np.array_equal(atlas[protected],original[protected]):raise ValueError('Observed non-hair pixels changed')
    Image.fromarray(atlas).save(output/'head-atlas.png')
    provenance=np.zeros(valid.shape,np.uint8);provenance[observed]=1;provenance[estimated]=2;provenance[changed_hair]=3
    Image.fromarray(provenance).save(output/'appearance-labels.png')
    export_texture(v,f,uv,atlas,output/'head.glb')
    report={'status':'ASSEMBLY_APPEARANCE_EXPORTED','policy':'ear_anchored_hairline_skin_completion_v1',
        'bundle_sha256':sha(bundle),'atlas_sha256':sha(output/'head-atlas.png'),
        'head_sha256':sha(output/'head.glb'),'ear_anchors':anchors,
        'observed_non_hair_pixels_preserved':True,'estimated_skin_texels':int(estimated.sum()),
        'designed_hair_texels':int(changed_hair.sum()),'hair_inferred_from_photos':False,
        'hidden_skin_is_estimated':True,'ready_for_machine':False}
    write(output/'report.json',report)
    return report
