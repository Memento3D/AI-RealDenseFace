"""Integrated review bundle: color/feature experiment plus hair/glasses/cap templates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import re
import base64
import io
from PIL import Image

import numpy as np
from export_workspace_glb import export
from export_workspace_glb import sample
from add_graduation_cap import load,save,append,design
from head_assets import short_hair,glasses


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True,help='Used to bind optional back/glasses references; not fitted by this script')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve()
    if out.exists():p.error('Choose a new output directory')
    html=a.workspace.read_text(encoding='utf-8-sig')
    match=re.search(r'<script id="data" type="application/json">(.*?)</script>',html,re.S)
    if not match:p.error('Expected head.workspace.html')
    payload=json.loads(match.group(1));inputs=[]
    for i,slot in enumerate(('front','left','right')):
        found=[x for x in a.input_dir.iterdir() if x.is_file() and x.stem.lower()==slot and x.suffix.lower() in ('.jpg','.jpeg','.png')]
        if len(found)!=1:p.error('Expected exactly one input photograph: '+slot)
        with Image.open(found[0]) as im:original=np.array(im.convert('RGB'))
        with Image.open(io.BytesIO(base64.b64decode(payload['images'][i].split(',',1)[1]))) as im:embedded=np.array(im.convert('RGB'))
        if not np.array_equal(original,embedded):p.error('Input photograph does not match workspace: '+slot)
        inputs.append(found[0])
    inputs += [x for x in a.input_dir.iterdir() if x.is_file() and x.stem.lower()=='back' and x.suffix.lower() in ('.jpg','.jpeg','.png')]
    if (a.input_dir/'references').is_dir():
        inputs += [x for x in (a.input_dir/'references').iterdir() if x.is_file() and x.stem.lower() in ('front-glasses','left-glasses','right-glasses') and x.suffix.lower() in ('.jpg','.jpeg','.png')]
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    report=dict(status='FAILED',geometry_modified=False,likeness_verified=False,printability_verified=False,
        source_workspace_sha256=hashlib.sha256(a.workspace.read_bytes()).hexdigest(),
        assets_origin='original procedural templates; fitted heuristically, not reconstructed from photos',
        reference_usage='hashes recorded for review only; back/glasses photos not automatically fitted',
        limitations=['Not true albedo recovery','No gaze/eyelid/jaw refitting','No lens optics','Hair is a short template, not individual hair reconstruction','No collision or physical-scale verification'])
    try:
        export(a.workspace,out/'head.skin.glb',32,True,True,True)
        doc,blob=load(out/'head.skin.glb')
        acc=doc['accessors'][0];view=doc['bufferViews'][acc['bufferView']]
        vertices=np.frombuffer(blob,dtype='<f4',count=acc['count']*3,offset=view.get('byteOffset',0)).reshape(-1,3).copy()
        points=np.unique(vertices,axis=0)
        er=json.loads((out/'head.skin.report.json').read_text())['eye_components']
        # Front hairline cue only: not semantic segmentation or 3D hair recovery.
        q=np.asarray(payload['coordinates'][0]);source=np.asarray(Image.open(io.BytesIO(base64.b64decode(payload['images'][0].split(',',1)[1]))).convert('RGB'))/255
        sampled=sample(source,q[:,:2]/q[:,2,None]);height=np.ptp(vertices[:,1]);width=np.ptp(vertices[:,0]);eye_y=np.mean(np.array(er['eye_centers_native'])[:,1])
        center_x=(vertices[:,0].min()+vertices[:,0].max())/2
        cue=(vertices[:,1]>eye_y+.06*height)&(np.abs(vertices[:,0]-center_x)<width*.23)&(vertices[:,2]>0)&(sampled.mean(1)<.24)
        front_line=None
        if cue.sum()>=10:
            front_line=float(np.clip(np.percentile(vertices[cue,1],10),eye_y+.07*height,vertices[:,1].max()-.08*height))
        report['hairline_cue']=dict(method='dark front-photo pixels above eyes; heuristic',samples=int(cue.sum()),front_y_native=front_line)
        hair=short_hair(points,front_line);frames=glasses(er['eye_centers_native'],points);cap,capfit=design(points)
        variants={
            'head.hair.glb':hair,
            'head.hair_glasses.glb':hair+frames,
            'head.hair_cap.glb':hair+cap,
            'head.complete.glb':hair+frames+cap}
        for name,parts in variants.items():
            target=copy.deepcopy(doc);data=bytearray(blob)
            for part,triangles,color in parts:append(target,data,part,triangles,color)
            save(out/name,target,data)
        refs={}
        for path in sorted(inputs):
            refs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        report.update(status='INTEGRATED_REVIEW_READY',reference_sha256=refs,
            variants=['head.skin.glb',*variants],cap_fit=capfit,
            color_report='head.skin.report.json',review_items=['Compare skin color with original front image','Inspect eye/mouth edges','Inspect hairline, ears and nape','Check glasses bridge/temples','Inspect cap fit with hair hidden and shown'])
    except Exception as e:
        report['error']=str(e)
        raise
    finally:
        report['elapsed_seconds']=round(time.perf_counter()-started,3)
        (out/'complete.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2));print('Output: '+str(out))


if __name__=='__main__':main()
