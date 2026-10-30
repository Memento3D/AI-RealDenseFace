"""Make shared benchmark inputs and preserve original photographic face bytes."""
import argparse,base64,hashlib,io,json,re,shutil
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def hair_canvas(image,labels,size=512):
    mask=labels==17
    if mask.sum()<100:raise ValueError('Too few hair pixels for a masked reconstruction input')
    y,x=np.nonzero(mask);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
    crop=np.asarray(image)[box[1]:box[3],box[0]:box[2]];alpha=mask[box[1]:box[3],box[0]:box[2]].astype('uint8')*255
    foreground=Image.fromarray(np.dstack([crop,alpha]));scale=size*.85/max(foreground.size)
    foreground=foreground.resize(tuple(max(1,round(v*scale)) for v in foreground.size),Image.Resampling.LANCZOS)
    rgba=Image.new('RGBA',(size,size),(128,128,128,0));rgba.alpha_composite(foreground,((size-foreground.width)//2,(size-foreground.height)//2))
    background=Image.new('RGBA',(size,size),(128,128,128,255));background.alpha_composite(rgba)
    return rgba,background.convert('RGB'),box

def prepare(project,workspace,mask_run,output):
    project=Path(project).resolve();workspace=Path(workspace).resolve();output=Path(output).resolve()
    digest=sha(workspace)
    if mask_run is None:
        choices=[]
        for f in (project/'runs').glob('*/report.json'):
            try:
                r=json.loads(f.read_text());folder=f.parent
                if r['face']['workspace_sha256']==digest and (folder/'shared/source_reference.glb').is_file() and (folder/'shared/view_0.labels.png').is_file():choices.append(folder)
            except (ValueError,KeyError,OSError):continue
        if not choices:
            reference=Path(__file__).resolve().parents[1]/'reference04'
            if (reference/'report.json').is_file() and json.loads((reference/'report.json').read_text())['face']['workspace_sha256']==digest:choices.append(reference)
        if not choices:raise ValueError('No matching source face + semantic masks. Pass -MaskRun to an existing V2.1/V2.2 result for this workspace.')
        mask_run=max(choices,key=lambda p:(p/'report.json').stat().st_mtime)
    mask_run=Path(mask_run).resolve();r=json.loads((mask_run/'report.json').read_text())
    if r['face']['workspace_sha256']!=digest:raise ValueError('Mask run belongs to a different workspace')
    labels_path=mask_run/'shared/view_0.labels.png';checks=json.loads((mask_run/'checksums.json').read_text())
    if sha(labels_path)!=checks['shared/view_0.labels.png']:raise ValueError('Mask checksum mismatch')
    source=mask_run/'shared/source_reference.glb'
    if sha(source)!=checks['shared/source_reference.glb']:raise ValueError('Original face checksum mismatch')
    match=re.search(r'<script id="data" type="application/json">(.*?)</script>',workspace.read_text(encoding='utf-8-sig'),re.S)
    if not match:raise ValueError('Workspace has no source data')
    data=json.loads(match.group(1));image=Image.open(io.BytesIO(base64.b64decode(data['images'][0].split(',',1)[1]))).convert('RGB');labels=np.asarray(Image.open(labels_path))
    if labels.shape!=np.asarray(image).shape[:2]:raise ValueError('Mask dimensions mismatch')
    output.mkdir(parents=True,exist_ok=False);image.save(output/'portrait.png');rgba,rgb,box=hair_canvas(image,labels)
    rgba.save(output/'hair_rgba.png');rgb.save(output/'hair_rgb.png');shutil.copy2(source,output/'original_face.glb')
    viewer=output/'viewer';(viewer/'assets/hair').mkdir(parents=True);web=Path(__file__).resolve().parents[1]/'assets/hair-studio'
    if not web.is_dir():raise ValueError('Bundled Hair Studio web assets are missing; re-extract the complete benchmark ZIP')
    shutil.copytree(web,viewer/'assets/hair-studio');run=viewer/'subject';(run/'shared').mkdir(parents=True)
    shutil.copy2(source,run/'shared/canonical_reference.glb') # Studio path contract only; bytes are original photographic face.
    (run/'report.json').write_text(json.dumps({'version':'HairBench original face','selected':None,'face_mode':'original photographic source, no canonical skin recoloring'}))
    shutil.copy2(mask_run/'profile.used.json',run/'profile.used.json')
    (run/'review.html').write_text('<meta charset="utf-8"><h1>Benchmark tóc — mặt nguồn giữ nguyên</h1><p>Các model mới chỉ tạo tóc; ghép khung trong Studio là preview, chưa kiểm tra in.</p>')
    manifest={'workspace_sha256':digest,'face_sha256':sha(source),'mask_sha256':sha(labels_path),'source_run':str(mask_run),'hair_crop_xyxy':box,
        'portrait':str(output/'portrait.png'),'hair_rgb':str(output/'hair_rgb.png'),'hair_rgba':str(output/'hair_rgba.png'),
        'face_reference':str(output/'original_face.glb'),'viewer_project':str(viewer),'viewer_run':str(run),
        'input_note':'DiffLocks uses original portrait; mesh baselines use hair-only mask on gray/transparent canvas. Conditioning differs and is recorded.',
        'geometry_or_skin_modified':False,'ready_for_machine':False}
    (output/'subject.json').write_text(json.dumps(manifest,indent=2));return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--mask-run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.project,a.workspace,a.mask_run,a.output),indent=2))
