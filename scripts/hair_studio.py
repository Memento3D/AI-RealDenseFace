"""Local Hair Studio: live asset inventory, cached preview fitting, unchanged head.

127.0.0.1 only. New STL/GLB files need no registry edit for preview. Unknown
styles use an explicitly approximate bounding-box fit, never print validation.
"""
import argparse,hashlib,json,mimetypes,os,re,sys,tempfile,threading,time,webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,unquote
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from add_graduation_cap import load,save
from figurine_preview import accessor,add_mesh
from figurine_geometry import close_boundaries
from figurine2_hair import fit_hair
from figurine2_render import srgb_to_linear

VERSION='HairStudio-1.0'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def confined(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Path outside permitted folder')
    return path

def validate_transform(value):
    result={k:float(value.get(k,default)) for k,default in [('scale',1.),('height_mm',0.),('yaw_deg',0.)]}
    if not all(np.isfinite(list(result.values()))) or not .5<=result['scale']<=1.5 or abs(result['height_mm'])>5 or abs(result['yaw_deg'])>180:
        raise ValueError('Preview controls outside bounds')
    return result

def transform_mesh(mesh,transform,pivot):
    t=validate_transform(transform);m=mesh.copy();angle=np.deg2rad(t['yaw_deg'])
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    m.vertices=(m.vertices-pivot)*t['scale']@rotation.T+pivot+np.array([0,t['height_mm'],0])
    return m

def mesh_glb(mesh,path,color=(35,28,24)):
    # Independent hair GLB, in meters, using the same material convention as V2.2.
    from run_pipeline_v2 import glb
    glb(path,[('HairPreview',mesh,np.r_[np.asarray(color)/255,1.])])

class Studio:
    def __init__(self,project,run,stable_seconds=.8):
        self.project=Path(project).resolve();self.run=Path(run).resolve();self.assets=self.project/'assets/hair'
        self.web=self.project/'assets/hair-studio';self.cache=self.run/'hair-studio/cache';self.cache.mkdir(parents=True,exist_ok=True)
        self.face_path=self.run/'shared/canonical_reference.glb'
        if not self.face_path.is_file():raise ValueError('Run needs shared/canonical_reference.glb from Pipeline V2.2')
        self.report=json.loads((self.run/'report.json').read_text(encoding='utf-8'))
        self.profile=json.loads((self.run/'profile.used.json').read_text(encoding='utf-8'))
        self.face_hash=sha(self.face_path);self.stable=stable_seconds;self.observed={};self.files={};self.jobs={};self.lock=threading.RLock()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='hair-preview')
        doc,blob=load(self.face_path);primitive=doc['meshes'][0]['primitives'][0]
        points=accessor(doc,blob,primitive['attributes']['POSITION'])*1000
        faces=accessor(doc,blob,primitive['indices']).reshape(-1,3).astype(int) if 'indices' in primitive else np.arange(len(points)).reshape(-1,3)
        head=trimesh.Trimesh(points,faces,process=True);head.remove_unreferenced_vertices();self.head,_=close_boundaries(head)
        centers=[]
        for m in doc['meshes'][1:3]:
            p=m['primitives'][0];v=accessor(doc,blob,p['attributes']['POSITION'])*1000
            if 'indices' in p:v=v[accessor(doc,blob,p['indices']).ravel().astype(int)]
            centers.append((v.min(0)+v.max(0))/2)
        self.eye_y=float(np.mean(centers,axis=0)[1]);self.base_y=float(self.head.bounds[0,1]-.1)
        upper=self.head.vertices[self.head.vertices[:,1]>self.eye_y+.07*self.head.extents[1]]
        lo,hi=np.quantile(upper,[.005,.995],axis=0)
        self.scalp_lo,self.scalp_hi=lo,hi;self.pivot=np.array([(lo[0]+hi[0])/2,self.eye_y,(lo[2]+hi[2])/2])
        self.current=None;selected=self.report.get('selected')
        if selected:
            candidate=self.run/'selected/candidate.json'
            if candidate.is_file():self.current=json.loads(candidate.read_text())
        # A persisted HTML copy provides useful instructions even when double-clicked.
        (self.run/'hair-studio').mkdir(exist_ok=True)
        if (self.web/'index.html').is_file():(self.run/'hair-studio.html').write_bytes((self.web/'index.html').read_bytes())

    def registry(self):
        path=self.assets/'library.json'
        try:return json.loads(path.read_text(encoding='utf-8-sig')).get('assets',[])
        except (OSError,ValueError):return []

    def inventory(self):
        result=[];seen=set();registry=self.registry();now=time.monotonic()
        for path in sorted(self.assets.rglob('*'),key=lambda p:p.as_posix().lower()):
            if path.suffix.lower() not in ['.stl','.glb'] or not path.is_file():continue
            try:
                relative=path.relative_to(self.assets).as_posix();safe=confined(self.assets,relative)
                stat=safe.stat();sidecar=safe.with_suffix('.hair.json')
                if sidecar.exists():confined(self.assets,sidecar.relative_to(self.assets))
                side_stat=sidecar.stat() if sidecar.exists() else None
                signature=(stat.st_size,stat.st_mtime_ns,side_stat.st_mtime_ns if side_stat else None,side_stat.st_size if side_stat else None)
                ident=hashlib.sha256(relative.encode()).hexdigest()[:20];seen.add(ident)
                old=self.observed.get(ident)
                if old is None or old['signature']!=signature:
                    self.observed[ident]={'signature':signature,'since':now,'digest':None}
                state=self.observed[ident]
                if now-state['since']<self.stable:
                    with self.lock:self.files.pop(ident,None)
                    result.append({'id':ident,'filename':relative,'name':safe.stem,'revision':str(signature),'state':'copying','fit':'Đang chờ chép file xong'});continue
                if stat.st_size>128*1024*1024:raise ValueError('Asset exceeds 128 MiB preview limit')
                if state['digest'] is None:state['digest']=sha(safe)
                digest=state['digest'];metadata=json.loads(sidecar.read_text(encoding='utf-8-sig')) if sidecar.exists() else {}
                if not isinstance(metadata,dict):raise ValueError('Sidecar must be a JSON object')
                registered=next((a for a in registry if a.get('sha256')==digest),None)
                mode='registered' if registered else 'approximate'
                if self.current and self.current['fit']['asset_sha256']==digest:mode='current'
                revision=hashlib.sha256((digest+json.dumps({'metadata':metadata,'registered':registered},sort_keys=True)).encode()).hexdigest()
                entry={'id':ident,'filename':relative,'name':str(metadata.get('name',registered.get('name') if registered else safe.stem))[:160],
                    'revision':revision,'state':'available','fit':{'current':'Tóc đã xuất trong kết quả V2.2','registered':'Mẫu có thông số khớp cavity','approximate':'Khớp khung sơ bộ — có thể cần chỉnh hướng/cỡ'}[mode],
                    'mode':mode,'sha256':digest,'path':safe,'metadata':metadata,'registered':registered}
                with self.lock:self.files[ident]=entry
                result.append({k:v for k,v in entry.items() if k not in ['path','metadata','registered']})
            except Exception as e:
                ident=hashlib.sha256(path.relative_to(self.assets).as_posix().encode()).hexdigest()[:20]
                with self.lock:self.files.pop(ident,None)
                result.append({'id':ident,'filename':path.name,'name':path.stem,'state':'error','error':str(e),'revision':'error'})
        with self.lock:
            for ident in list(self.files):
                if ident not in seen:self.files.pop(ident,None)
        return result

    def request(self,ident,revision):
        self.inventory()
        with self.lock:
            if ident not in self.files:raise ValueError('Asset missing or still copying; refresh the list')
            entry=dict(self.files[ident])
            if revision!=entry['revision']:raise ValueError('Asset changed; refresh before loading')
            key=hashlib.sha256((VERSION+self.face_hash+entry['revision']+ident).encode()).hexdigest()[:32]
            if key not in self.jobs:
                self.jobs[key]={'id':key,'state':'queued','asset_id':ident,'revision':revision,'name':entry['name']}
                self.executor.submit(self.build,key,entry)
            return dict(self.jobs[key])

    def build(self,key,entry):
        try:
            with self.lock:self.jobs[key]['state']='loading'
            path=entry['path'];mode=entry['mode'];metadata=entry['metadata']
            if sha(path)!=entry['sha256']:raise ValueError('Asset changed during loading; select it again after copy finishes')
            if mode=='current':
                mesh=trimesh.load(self.run/'selected/hair_mm.stl',force='mesh',process=True)
                v=mesh.vertices.copy();mesh.vertices=np.c_[v[:,0],v[:,2]+self.base_y,-v[:,1]]
            elif mode=='registered':
                asset=dict(entry['registered']);asset['path']=path.relative_to(self.project).as_posix()
                mesh,_=fit_hair(asset,self.project,self.head,self.eye_y,self.profile['fit'],'cage')
            else:
                if path.suffix.lower()=='.glb':
                    d,_=load(path)
                    if any(b.get('uri') and not b['uri'].startswith('data:') for b in d.get('buffers',[])):raise ValueError('GLB preview requires embedded buffers')
                    scene=trimesh.load(path,force='scene',process=False,skip_materials=True)
                    mesh=scene.dump(concatenate=True)
                else:mesh=trimesh.load(path,force='mesh',process=True)
                if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):raise ValueError('Asset has no triangle mesh')
                up=metadata.get('up_axis','Y' if path.suffix.lower()=='.glb' else 'Z').upper()
                if up=='Z':mesh.vertices=mesh.vertices[:,[0,2,1]]*np.array([1,1,-1])
                elif up!='Y':raise ValueError('up_axis must be Y or Z')
                if not np.isfinite(mesh.vertices).all() or np.min(mesh.extents)<1e-7:raise ValueError('Degenerate or non-finite asset')
                width=float(self.scalp_hi[0]-self.scalp_lo[0]+1.0);factor=width/mesh.extents[0]
                mesh.vertices*=factor
                center=(mesh.bounds[0]+mesh.bounds[1])/2
                mesh.vertices+=np.array([self.pivot[0]-center[0],self.head.bounds[1,1]+.7-mesh.bounds[1,1],self.pivot[2]-center[2]])
            if len(mesh.faces)>500000:raise ValueError('Asset exceeds 500,000 triangles; simplify this source asset once')
            controls=validate_transform(metadata.get('transform',{}));mesh=transform_mesh(mesh,controls,self.pivot)
            dest=self.cache/(key+'.glb');temporary=self.cache/(key+'.tmp.glb')
            mesh_glb(mesh,temporary);temporary.replace(dest)
            # Keep fit geometry for GLB preview export; no head coordinates are altered.
            mesh.export(self.cache/(key+'.ply'))
            with self.lock:self.jobs[key].update(state='ready',model_url='/model/hair/'+key+'.glb',mode=mode,preview_only=True,
                ready_for_machine=False,triangles=len(mesh.faces),fit=entry['fit'])
        except Exception as exc:
            with self.lock:self.jobs[key].update(state='error',error=str(exc))

    def status(self,key):
        with self.lock:
            if key not in self.jobs:raise ValueError('Unknown job')
            return dict(self.jobs[key])

    def export(self,request):
        key=request.get('job','');job=self.status(key)
        if job['state']!='ready':raise ValueError('Hair is not ready')
        transform=validate_transform(request.get('transform',{}));hair=trimesh.load(self.cache/(key+'.ply'),force='mesh',process=False)
        hair=transform_mesh(hair,transform,self.pivot)
        doc,blob=load(self.face_path);color=np.r_[srgb_to_linear(np.array([35,28,24])/255),1.]
        add_mesh(doc,blob,'HairStudio_Preview',hair,np.tile(color,(len(hair.vertices),1)))
        doc['asset']['extras']={'studio':VERSION,'source_head_sha256':self.face_hash,'preview_only':True,'ready_for_machine':False,
            'asset':job['asset_id'],'revision':job['revision'],'transform':transform}
        # Original head buffers are shared unchanged. Only a new hair primitive is appended.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'preview.glb';save(p,doc,blob);return p.read_bytes()

    def close(self):self.executor.shutdown(wait=True,cancel_futures=True)


def handler(studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):pass
        def send(self,code,data,kind='application/json; charset=utf-8'):
            if isinstance(data,(dict,list)):data=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def allowed(self):
            port=self.server.server_address[1];return self.headers.get('Host') in [f'127.0.0.1:{port}',f'localhost:{port}']
        def do_GET(self):
            try:
                if not self.allowed():self.send(403,{'error':'Loopback Host required'});return
                path=unquote(urlparse(self.path).path)
                if path=='/api/assets':self.send(200,{'assets':studio.inventory()});return
                if path=='/api/info':
                    self.send(200,{'version':VERSION,'run':studio.run.name,'head_sha256':studio.face_hash,'head_url':'/model/head.glb',
                        'pivot_m':(studio.pivot/1000).tolist(),'head_height_mm':float(studio.head.extents[1]),'asset_folder':'assets/hair','poll_ms':2500});return
                if path.startswith('/api/jobs/'):
                    self.send(200,studio.status(path.rsplit('/',1)[1]));return
                if path=='/model/head.glb':self.send(200,studio.face_path.read_bytes(),'model/gltf-binary');return
                if path.startswith('/model/hair/'):
                    name=path.rsplit('/',1)[1]
                    if not re.fullmatch(r'[a-f0-9]{32}\.glb',name):raise ValueError('Invalid model ID')
                    self.send(200,confined(studio.cache,name).read_bytes(),'model/gltf-binary');return
                if path.startswith('/vendor/'):
                    p=confined(studio.web/'vendor',path[len('/vendor/'):])
                elif path.startswith('/review/'):
                    p=confined(studio.run,path[len('/review/'):])
                elif path in ['/','/hair-studio.html']:p=studio.web/'index.html'
                elif path in ['/studio.mjs','/selection.mjs']:p=studio.web/path[1:]
                else:self.send(404,{'error':'Not found'});return
                kind='text/javascript' if p.suffix in ['.js','.mjs'] else mimetypes.guess_type(p.name)[0] or 'application/octet-stream'
                self.send(200,p.read_bytes(),kind)
            except FileNotFoundError:self.send(404,{'error':'File not found'})
            except (ValueError,KeyError) as exc:self.send(400,{'error':str(exc)})
            except Exception as exc:self.send(500,{'error':str(exc)})
        def do_POST(self):
            try:
                port=self.server.server_address[1]
                if not self.allowed() or self.headers.get('Origin') not in [None,f'http://127.0.0.1:{port}',f'http://localhost:{port}']:
                    self.send(403,{'error':'Same-origin loopback request required'});return
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('Invalid JSON request size')
                value=json.loads(self.rfile.read(length));path=urlparse(self.path).path
                if path=='/api/load':self.send(200,studio.request(str(value['id']),str(value['revision'])))
                elif path=='/api/export':
                    raw=studio.export(value);self.send_response(200);self.send_header('Content-Type','model/gltf-binary');self.send_header('Content-Disposition','attachment; filename="hair-studio-preview.glb"');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
                else:self.send(404,{'error':'Not found'})
            except (ValueError,KeyError,TypeError) as exc:self.send(400,{'error':str(exc)})
            except Exception as exc:self.send(500,{'error':str(exc)})
    return Handler


def latest_run(project):
    for base in [Path(project)/'runs',Path(project)]:
        candidates=[]
        if not base.exists():continue
        for report in base.glob('*/report.json'):
            run=report.parent
            if not (run/'shared/canonical_reference.glb').is_file():continue
            try:
                data=json.loads(report.read_text())
                if data.get('selected') and (run/'selected/hair_mm.stl').is_file():candidates.append(run)
            except (OSError,ValueError):continue
        if candidates:return max(candidates,key=lambda p:(p/'report.json').stat().st_mtime)
    raise ValueError('No V2.2 run found. Run RUN-PIPELINE-V22.ps1 first, or pass -RunDir to extracted Results04 folder.')



def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=ROOT);p.add_argument('--run',type=Path);p.add_argument('--port',type=int,default=8765);p.add_argument('--open-browser',action='store_true')
    a=p.parse_args();studio=Studio(a.project,a.run or latest_run(a.project))
    try:server=ThreadingHTTPServer(('127.0.0.1',a.port),handler(studio))
    except OSError:
        if a.port==0:raise
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(studio))
    url='http://127.0.0.1:'+str(server.server_address[1])+'/hair-studio.html'
    print('Hair Studio: '+url,flush=True);print('Assets: '+str(studio.assets),flush=True);print('Keep this terminal open. Ctrl+C stops the local server.',flush=True)
    if a.open_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();studio.close()
if __name__=='__main__':main()
