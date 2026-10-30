"""Submit an existing input folder without PowerShell multipart-version assumptions."""
import argparse,json,time,urllib.request,uuid
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--assemble-job')
    p.add_argument('--url',default='http://127.0.0.1:8090');p.add_argument('--mode',choices=['mvp','components','raw'],default='mvp')
    p.add_argument('--views',choices=['front','front-left','front-right','all'],default='all')
    p.add_argument('--resolution',type=int,choices=[128,192,256],default=128);p.add_argument('--wait',action='store_true');a=p.parse_args()
    if a.assemble_job:
        request=urllib.request.Request(a.url+'/v1/head/jobs/'+a.assemble_job+'/assemble',data=b'')
    else:
        if not a.input:p.error('--input is required unless --assemble-job is given')
        allowed=['front']+(['left','right'] if a.views=='all' else a.views.split('-')[1:])
        if a.mode=='raw':allowed=['front']
        boundary=uuid.uuid4().hex;parts=[]
        for slot in allowed:
            paths=[f for f in a.input.iterdir() if f.is_file() and f.stem.lower()==slot and f.suffix.lower() in ('.png','.jpg','.jpeg','.webp')]
            if not paths and slot!='front':continue
            if len(paths)!=1:p.error('Need exactly one image for '+slot)
            content=paths[0].read_bytes()
            if len(content)>12*1024*1024:p.error(slot+' exceeds 12 MiB')
            parts.append(('--'+boundary+'\r\nContent-Disposition: form-data; name="'+slot+'"; filename="'+slot+'.image"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+content+b'\r\n')
        for key,value in [('mode',a.mode),('resolution',str(a.resolution))]:
            parts.append(('--'+boundary+'\r\nContent-Disposition: form-data; name="'+key+'"\r\n\r\n'+value+'\r\n').encode())
        parts.append(('--'+boundary+'--\r\n').encode())
        request=urllib.request.Request(a.url+'/v1/head/jobs',data=b''.join(parts),headers={'Content-Type':'multipart/form-data; boundary='+boundary})
    with urllib.request.urlopen(request,timeout=60) as r:result=json.load(r)
    print(json.dumps(result,indent=2),flush=True)
    if a.wait:
        deadline=time.monotonic()+10800
        while time.monotonic()<deadline:
            with urllib.request.urlopen(a.url+result['status_url'],timeout=30) as r:row=json.load(r)
            print(row['status'],row.get('current_stage'),flush=True)
            if row['status'] not in ('RUNNING','QUEUED'):
                print(json.dumps(row,indent=2));return 0 if row['status'] in ('MVP_ASSEMBLED','RAW_EXPORTED','COMPONENTS_EXPORTED') else 2
            time.sleep(2)
        raise TimeoutError('Polling deadline reached; job was not cancelled; check its status URL')
    return 0

if __name__=='__main__':raise SystemExit(main())
