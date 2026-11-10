"""Read-only inventory: protect runtime references before deleting generated runs."""
from pathlib import Path
import os,json,re,subprocess,collections

root=Path(__file__).resolve().parents[2]
destination=root/'maintenance/local-cleanup/20261110'
destination.mkdir(parents=True,exist_ok=True)
protected=set()
reasons=collections.defaultdict(list)
def protect(path,reason):
    try:
        p=path.resolve();p.relative_to(root)
    except (ValueError,OSError):return
    if p.is_file():
        protected.add(p);reasons[str(p.relative_to(root))].append(reason)
tracked=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
for relative in tracked:
    p=root/relative
    protect(p,'tracked source')
    if p.suffix.lower() not in {'.json','.py','.ps1','.toml','.yaml','.yml'} or not p.exists():continue
    if p.stat().st_size>2*1024**2:continue
    text=p.read_text(encoding='utf-8-sig',errors='replace')
    for value in re.findall(r'''["']((?:runs[/\\])[^"'\r\n]+)["']'''.replace('\\\\r','\\r').replace('\\\\n','\\n'),text):
        target=root/value.replace('\\','/')
        if target.is_file():protect(target,'literal reference: '+relative)
        # Keep entire explicit local environment directories referenced by scripts.
        elif target.is_dir() and (any(x.startswith('.venv') for x in target.parts) or
                                  (relative.startswith('configs/') and target.name == 'src')):
            for folder,dirs,files in os.walk(target,followlinks=False):
                for name in files:protect(Path(folder)/name,'referenced runtime environment/source')
for config in (root/'configs').glob('*.json'):
    try:data=json.loads(config.read_text(encoding='utf-8-sig'))
    except (ValueError,OSError):continue
    jobs=data.get('paths',{}).get('jobs')
    if not isinstance(jobs,str):continue
    base=root/jobs
    if base.is_dir():
        children=sorted((p for p in base.iterdir() if p.is_dir()),key=lambda p:p.stat().st_mtime,reverse=True)
        for child in children[:1]:
            for folder,dirs,files in os.walk(child,followlinks=False):
                for name in files:protect(Path(folder)/name,'latest API job retained')
# Protect all local runtime environments, vendor and actual assets inside runs.
files=[];groups=collections.defaultdict(lambda:[0,0]);parameters=[];remove=[]
important={'.json','.yaml','.yml','.toml','.ini','.cfg','.csv','.tsv','.md','.txt','.npz','.npy','.pkl'}
for folder,dirs,names in os.walk(root/'runs',followlinks=False):
    dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
    for name in names:
        p=Path(folder)/name
        if p.is_symlink():continue
        try:p.resolve().relative_to(root)
        except ValueError:continue
        stat=p.stat();relative=p.relative_to(root).as_posix();parts=p.relative_to(root/'runs').parts
        groups[parts[0]][0]+=1;groups[parts[0]][1]+=stat.st_size
        if any(x.startswith('.venv') or x in {'site-packages','vendor','models','weights','node_modules','python','.git','Lib','Scripts','include'} for x in parts):
            protect(p,'runtime/dependency inside runs')
        if p.suffix.lower() in {'.py','.ps1','.sh','.bat','.cmd'}:protect(p,'experiment source retained')
        if p in protected:continue
        if p.suffix.lower() in important and stat.st_size<=32*1024**2:
            parameters.append(dict(path=relative,size=stat.st_size))
        else:remove.append(dict(path=relative,size=stat.st_size))
plan=dict(root=str(root),protected={str(p.relative_to(root)):reasons[str(p.relative_to(root))] for p in protected if 'runs' in p.parts},
    preserve_parameters=parameters,delete_generated=remove,
    groups=dict(sorted(groups.items(),key=lambda x:x[1][1],reverse=True)))
(destination/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(dict(delete_files=len(remove),delete_gib=round(sum(x['size'] for x in remove)/2**30,3),
    parameter_files=len(parameters),parameter_gib=round(sum(x['size'] for x in parameters)/2**30,3),
    protected_runs_files=len(plan['protected']),largest_groups=list(plan['groups'].items())[:15]),indent=2))
