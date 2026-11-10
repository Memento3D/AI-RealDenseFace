"""Blender-only safe inspection of downloaded source assets."""
import argparse,json,sys,hashlib
from pathlib import Path

def main():
    import bpy
    from mathutils import Vector
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.root/'manifest.json').read_text())
    rows=[]
    for entry in manifest['assets']:
        source=a.root/entry['original']
        if source.suffix=='.zip':source=next((source.parent/'unpacked').glob('*.blend'))
        bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
        objects=[]
        for obj in bpy.context.scene.objects:
            if obj.type!='MESH':continue
            points=[obj.matrix_world@Vector(c) for c in obj.bound_box]
            objects.append({'name':obj.name,'vertices':len(obj.data.vertices),'polygons':len(obj.data.polygons),
                'bounds':[[min(v[k] for v in points) for k in range(3)],[max(v[k] for v in points) for k in range(3)]],
                'materials':[m.name if m else None for m in obj.data.materials],
                'modifiers':[(m.name,m.type) for m in obj.modifiers]})
        rows.append({'id':entry['id'],'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'objects':objects})
        print(entry['id'],[(o['name'],o['vertices'],o['polygons']) for o in objects],flush=True)
    (a.output/'inspection.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':main()
