"""Deterministic input discovery, content binding and verified stage reuse."""
import hashlib,json,re,shutil,time
from pathlib import Path
from datetime import datetime,timezone

SLOTS=('front','left','right');EXTENSIONS={'.jpg','.jpeg','.png','.webp'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def stamp():return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')

def dataset_ids(value):
 result=[]
 for token in re.split('[,; ]+',value or ''):
  if not token:continue
  match=re.fullmatch(r'(\d+)-(\d+)',token)
  if match:
   a,b=map(int,match.groups())
   if a>b or b-a>99:raise ValueError('Dataset range must ascend and contain at most100 entries')
   width=max(len(match[1]),len(match[2]));names=[str(i).zfill(width) for i in range(a,b+1)]
  else:names=[token]
  for name in names:
   if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',name):raise ValueError('Unsafe dataset ID: '+name)
   if name not in result:result.append(name)
 return result

def discover(root,ids=None,single=None):
 root=Path(root).resolve()
 if single is not None:
  folder=Path(single).resolve();return [(folder.name,folder)]
 names=dataset_ids(ids) if ids else sorted(p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink())
 return [(name,root/name) for name in names]

def inspect_dataset(dataset_id,folder):
 from PIL import Image,ImageOps
 import numpy as np
 folder=Path(folder).resolve();result={'id':dataset_id,'input_dir':str(folder),'status':'INPUT_INVALID','views':{},'errors':[],'warnings':[]}
 if not folder.is_dir():result['errors'].append('DATASET_DIRECTORY_MISSING');return result
 mapping=read(folder/'headlab.json').get('views',{}) if (folder/'headlab.json').is_file() else {}
 for slot in (*SLOTS,'back'):
  if slot in mapping:
   image_path=(folder/mapping[slot]).resolve()
   if not image_path.is_relative_to(folder):raise ValueError('Mapped input escapes dataset directory')
   found=[image_path] if image_path.is_file() else []
  else:found=[p for p in folder.iterdir() if p.is_file() and p.stem.lower()==slot and p.suffix.lower() in EXTENSIONS]
  if len(found)!=1:
   if slot in SLOTS:result['errors'].append(slot+':EXPECTED_EXACTLY_ONE_IMAGE')
   elif found:result['errors'].append('back:AMBIGUOUS_IMAGE')
   continue
  p=found[0]
  if p.is_symlink():raise ValueError('Symlink input not allowed')
  try:
   with Image.open(p) as im:
    if im.width*im.height>24_000_000:raise ValueError('IMAGE_PIXEL_BUDGET_EXCEEDED')
    raw_size=im.size;orientation=im.getexif().get(274,1);im=ImageOps.exif_transpose(im).convert('RGB');size=im.size;small=im.copy();small.thumbnail((512,512));rgb=np.asarray(small,float)/255
   light=rgb@np.array([.2126,.7152,.0722]);row={'path':str(p),'sha256':sha(p),'raw_size':list(raw_size),'oriented_size':list(size),'exif_orientation':orientation,'whole_image_luma_median':float(np.median(light)),'whole_image_clipped_bright_fraction':float((rgb.max(2)>.98).mean()),'scope':'whole image only; not face quality or recovered skin tone'}
   result['views'][slot]=row
  except Exception as e:result['errors'].append(slot+':'+str(e))
 if len({r['sha256'] for k,r in result['views'].items() if k in SLOTS})<len([k for k in result['views'] if k in SLOTS]):result['errors'].append('DUPLICATE_REQUIRED_VIEW_CONTENT')
 result['input_fingerprint']=key({'dataset_id':dataset_id,'views':{k:r['sha256'] for k,r in result['views'].items()}})
 result['status']='INPUTS_DISCOVERED' if not result['errors'] else 'INPUT_INVALID'
 if 'back' in result['views']:result['warnings'].append('BACK_AVAILABLE_UNCALIBRATED: retained for review; not falsely treated as a fitted camera')
 return result

def seal_stage(folder,stage_key):
 folder=Path(folder);rows={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='stage.binding.json'}
 write(folder/'stage.binding.json',{'key':stage_key,'files':rows})

def reuse_stage(source,destination,stage_key):
 source=Path(source);destination=Path(destination);binding=source/'stage.binding.json'
 if not binding.is_file():return False
 data=read(binding)
 if data.get('key')!=stage_key:return False
 if not data.get('files'):raise ValueError('EMPTY_STAGE_BINDING')
 for rel,digest in data['files'].items():
  path=source/rel
  if not path.resolve().is_relative_to(source.resolve()) or path.is_symlink() or not path.is_file() or sha(path)!=digest:raise ValueError('STALE_OR_TAMPERED_STAGE: '+rel)
 if destination.exists():raise ValueError('Refusing overwrite while reusing stage')
 shutil.copytree(source,destination);return True

def verify_code(root,manifest):
 for row in read(manifest)['files']:
  path=Path(root)/row['path']
  if not path.is_file() or sha(path)!=row['sha256']:raise ValueError('Code/config checksum mismatch: '+row['path'])
