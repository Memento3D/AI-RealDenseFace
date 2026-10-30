"""Fetch one pinned upstream source archive; never overwrite an existing vendor tree."""
import io,zipfile,urllib.request,shutil
from pathlib import Path
from .common import ROOT,CODE_REVISION,sha,write,check_repo,source_repo

def run():
 repo=source_repo(ROOT)
 if repo is not None:print('Using verified upstream source in place:',repo);return repo
 if shutil.disk_usage(ROOT).free<10*1024**3:raise ValueError('Planning guard: need 10GiB free on project drive before creating the environment')
 url='https://codeload.github.com/VAST-AI-Research/TripoSR/zip/'+CODE_REVISION
 data=urllib.request.urlopen(url,timeout=120).read();staging=ROOT/'vendor/TripoSR-probe-source'
 if staging.exists():raise ValueError('Previous incomplete source download exists: '+str(staging))
 staging.mkdir(parents=True)
 with zipfile.ZipFile(io.BytesIO(data)) as archive:
  for entry in archive.infolist():
   parts=Path(entry.filename).parts[1:]
   if not parts or entry.is_dir():continue
   destination=staging.joinpath(*parts)
   if not destination.resolve().is_relative_to(staging.resolve()):raise ValueError('Unsafe source archive path')
   destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(archive.read(entry))
 files={p.relative_to(staging).as_posix():sha(p) for p in staging.rglob('*') if p.is_file()}
 write(staging/'probe.source.json',{'repo':'https://github.com/VAST-AI-Research/TripoSR','revision':CODE_REVISION,'archive_sha256':__import__('hashlib').sha256(data).hexdigest(),'files':files})
 check_repo(staging);print('Pinned upstream source ready; no directory rename required:',staging);return staging
if __name__=='__main__':run()
