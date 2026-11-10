import os, hashlib, json, collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]; runs=root/'runs'
groups=collections.defaultdict(list); totals=collections.Counter()
excluded={'.git','.venv','.venv-mica','python','src','weights','uv-cache'}
for base,dirs,files in os.walk(runs):
    dirs[:]=[d for d in dirs if d not in excluded and not Path(base,d).is_symlink()]
    for name in files:
        p=Path(base,name)
        if p.is_symlink():continue
        size=p.stat().st_size;totals[str(p.relative_to(runs).parts[0])]+=size
        if size>=8*1024**2:groups[size].append(p)
duplicates=[]
for size,paths in groups.items():
    if len(paths)<2:continue
    hashes=collections.defaultdict(list)
    for p in paths:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        hashes[h.hexdigest()].append(str(p.relative_to(root)))
    for h,ps in hashes.items():
        if len(ps)>1:duplicates.append({'bytes':size,'sha256':h,'paths':ps})
out={'scope':'runs excluding environments, weights and vendored source','totals':totals,'duplicates':sorted(duplicates,key=lambda x:x['bytes']*(len(x['paths'])-1),reverse=True)}
(root/'maintenance/run-duplicates-20261104.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out['duplicates'],indent=2))
