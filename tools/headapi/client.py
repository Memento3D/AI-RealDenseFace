"""Submit an existing input folder without PowerShell multipart-version assumptions."""
import argparse,json,os,time,urllib.request,uuid
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--url',default='http://127.0.0.1:8090');p.add_argument('--wait',action='store_true');a=p.parse_args()
    api_key=os.environ.get('HEADAPI_API_KEY')
    if not api_key:p.error('Set HEADAPI_API_KEY before submitting to the authenticated API')
    boundary=uuid.uuid4().hex;parts=[]
    for slot in ('front','left','right','back'):
        paths=[f for f in a.input.iterdir() if f.is_file() and f.stem.lower()==slot and f.suffix.lower() in ('.png','.jpg','.jpeg','.webp')]
        if slot=='back' and not paths:continue
        if len(paths)!=1:p.error('Need exactly one image for '+slot)
        content=paths[0].read_bytes()
        if len(content)>12*1024*1024:p.error(slot+' exceeds 12 MiB')
        parts.append(('--'+boundary+'\r\nContent-Disposition: form-data; name="'+slot+'"; filename="'+slot+'.image"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+content+b'\r\n')
    parts.append(('--'+boundary+'--\r\n').encode())
    headers={'Content-Type':'multipart/form-data; boundary='+boundary,'X-API-Key':api_key}
    request=urllib.request.Request(a.url+'/v1/head/jobs',data=b''.join(parts),headers=headers)
    with urllib.request.urlopen(request,timeout=60) as r:result=json.load(r)
    print(json.dumps(result,indent=2),flush=True)
    if a.wait:
        deadline=time.monotonic()+10800
        while time.monotonic()<deadline:
            status_request=urllib.request.Request(a.url+result['status_url'],headers={'X-API-Key':api_key})
            with urllib.request.urlopen(status_request,timeout=30) as r:row=json.load(r)
            print(row['status'],row.get('current_stage'),flush=True)
            if row['status'] not in ('RUNNING','QUEUED'):
                print(json.dumps(row,indent=2));return 0 if row['status']=='MVP_EXPORTED' else 2
            time.sleep(2)
        raise TimeoutError('Polling deadline reached; job was not cancelled; check its status URL')
    return 0

if __name__=='__main__':raise SystemExit(main())
