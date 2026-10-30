"""Build a source workspace with explicit, view-specific camera/geometry contracts."""
import argparse,base64,io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from head24_policy import validate_camera_contract,InputRejected,project
from preview_rdf_three_textures import visible_faces
from memento3d_head.adapters.head_texture_workspace import digest
from memento3d_head.adapters.smooth_preview import vertex_normals

def texture_weights(covered,facing):
    covered=np.asarray(covered,dtype=bool)
    facing=np.asarray(facing,dtype=float)
    if covered.ndim!=2 or covered.shape[0]!=3 or facing.shape!=covered.shape:
        raise ValueError('Expected three view arrays with one value per triangle')
    if not np.isfinite(facing).all():raise ValueError('Nonfinite triangle facing')
    per_view=covered*np.maximum(facing-.35,0)**2
    weights=np.repeat(per_view.T[:,None,:],3,axis=1)
    front=covered[0]
    weights[front,:,1:]=0
    weights[front,:,0]*=4
    hard=np.zeros_like(weights)
    chosen=np.argmax(weights.sum(1),axis=1)
    active=weights.sum((1,2))>0
    for i in range(3):hard[(chosen==i)&active,:,i]=1
    return weights,hard


def run(fit,crops,out):
    report0=json.loads((fit/'report.json').read_text());assert report0['status']=='FIT_GATES_PASSED'
    if digest(fit/'projection_bundle.npz')!=report0['projection_bundle_sha256']:raise ValueError('Projection bundle checksum mismatch')
    if digest(crops/'report.json')!=report0['crop_report_sha256']:raise ValueError('Crop metadata mismatch')
    crop_report=json.loads((crops/'report.json').read_text())
    for slot in ('front','left','right'):
        if digest(crops/(slot+'.png'))!=crop_report['views'][slot]['crop_sha256']:raise ValueError('Crop image changed: '+slot)
    with np.load(fit/'projection_bundle.npz',allow_pickle=False) as b:
        v=b['vertices'];faces=b['faces'];local=b['view_vertices'];cameras=b['cameras'];cam=b['camera_points'];K=b['intrinsics']
    coords=[];covered=[];facing=[];images=[];checks=[]
    for i,slot in enumerate(('front','left','right')):
        q,valid,angles=visible_faces(cam[i],faces,K[i],(1024,1024),None)
        checks.append(validate_camera_contract(local[i],q,cameras[i],(1024,1024)))
        coords.append(q);covered.append(valid);facing.append(angles)
        stream=io.BytesIO();Image.open(crops/(slot+'.png')).convert('RGB').save(stream,format='PNG')
        images.append('data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())
    # Preserve true expression-specific projective sampling for texture.
    weights,hard=texture_weights(covered,facing)
    if np.count_nonzero(weights.sum((1,2)))<100:raise InputRejected('INSUFFICIENT_VISIBLE_FACE_TRIANGLES')
    center=(v.min(0)+v.max(0))/2
    report={'status':'TEXTURE_REVIEW_READY','pipeline_type':'head24_explicit_camera_workspace','input_sha256':{str(fit/'projection_bundle.npz'):digest(fit/'projection_bundle.npz'),str(crops/'report.json'):digest(crops/'report.json')},'camera_checks':checks,'geometry_modified':False,'likeness_verified':False,'printability_verified':False,'note':'coordinates belong to their own view expression; source geometry uses front expression'}
    payload={'vertices':v[faces].reshape(-1,3).tolist(),'normals':vertex_normals(v,faces)[faces].reshape(-1,3).tolist(),'coordinates':[q[faces].reshape(-1,3).tolist() for q in coords],'weights':weights.reshape(-1,3).tolist(),'hard':hard.reshape(-1,3).tolist(),'images':images,'gains':[[1,1,1]]*3,'skin':[.65,.48,.4],'center':center.tolist(),'radius':float(np.linalg.norm(v-center,axis=1).max()),'report':report,'camera_contract':{'version':1,'units':'native model units; pixel camera; image origin top-left','cameras':cameras.tolist(),'view_vertices':[p[faces].reshape(-1,3).tolist() for p in local]}}
    front_meta=crop_report['views']['front']
    if front_meta.get('landmarks_crop') is not None:
        import hashlib
        with Image.open(crops/'front.png') as source_image:
            rgb=np.asarray(source_image.convert('RGB'),dtype=np.uint8)
        payload['front_observation']={'landmarks_crop':front_meta['landmarks_crop'],'crop_pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'crop_file_sha256':front_meta['crop_sha256'],'crop_metadata_sha256':digest(crops/'report.json'),'method':'image-bound MediaPipe landmarks; not ground truth or hidden-anatomy recovery'}
    out.mkdir(parents=True,exist_ok=False)
    template=(ROOT/'src/memento3d_head/adapters/head_texture_workspace.html').read_text(encoding='utf-8')
    encoded=json.dumps(payload,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (out/'head.workspace.html').write_text(template.replace('__WORKSPACE_DATA__',encoded),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Workspace:',out/'head.workspace.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--crops',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.fit,a.crops,a.output)
