"""One-time controlled registration of a new closed hair asset.

Default is inspect/dry-run. --register adds a new ID, never replaces an existing
one. Cavity anchors must be prepared by an asset author, not guessed per order.
"""
import argparse,json,re,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import trimesh
from figurine2_face import sha
from figurine2_hair import read_library

ROTATIONS={'zup_front_minus_y':[[1,0,0],[0,0,1],[0,-1,0]],
           'zup_front_plus_y':[[-1,0,0],[0,0,1],[0,1,0]],
           'yup_front_plus_z':[[1,0,0],[0,1,0],[0,0,1]],
           'yup_front_minus_z':[[-1,0,0],[0,1,0],[0,0,-1]]}

def register(source,asset_id,name,orientation,roof,radius,base,register=False,root=ROOT):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}',asset_id):raise ValueError('Asset ID must contain letters/numbers/_/- only')
    registry=root/'assets/hair/library.json';library=json.loads(registry.read_text(encoding='utf-8-sig'))
    if any(a['id']==asset_id for a in library['assets']):raise ValueError('Asset ID already exists; use a new versioned ID')
    mesh=trimesh.load(source,force='mesh',process=True)
    if not mesh.is_volume or len(mesh.split())!=1:raise ValueError('Asset must be one closed oriented positive volume')
    if orientation not in ROTATIONS or not np.isfinite([roof,radius,base]).all() or roof<=base or not 1<radius<100:raise ValueError('Invalid orientation/cavity anchors')
    record={'id':asset_id,'name':name,'path':'assets/hair/'+asset_id+'/source.stl','sha256':sha(source),
            'source':str(Path(source).name),'license_status':'record source terms during asset preparation',
            'input_units':'mm assumed; author must verify scale','source_to_y_up':ROTATIONS[orientation],
            'inner_roof_y':roof,'inner_radius':radius,'source_base_y':base,'front_direction':[0,0,1],
            'limits':{'scale_min':.65,'scale_max':2.,'cage_displacement_mm':1.25},
            'custom':{'volume':0.,'fringe_lift_mm':0.,'nape_lift_mm':0.,'yaw_deg':0.},
            'validation_status':'IMPORTED_NOT_FIT_OR_PRINT_VALIDATED'}
    report={'registered':False,'record':record,'triangles':len(mesh.faces),'bounds_in_input_units':mesh.bounds.tolist(),
            'next':'check canonical front/cavity anchors, fit to reference and varied heads, then validate STL and print profile'}
    if register:
        destination=root/record['path']
        if destination.exists():raise ValueError('Asset destination exists; refusing overwrite')
        destination.parent.mkdir(parents=True);shutil.copy2(source,destination)
        library['assets'].append(record)
        temporary=registry.with_suffix('.json.tmp');temporary.write_text(json.dumps(library,indent=2),encoding='utf-8')
        read_library(temporary,root);temporary.replace(registry);report['registered']=True
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--id',required=True);p.add_argument('--name',required=True);p.add_argument('--orientation',choices=ROTATIONS,required=True)
    p.add_argument('--inner-roof-y',type=float,required=True);p.add_argument('--inner-radius',type=float,required=True);p.add_argument('--base-y',type=float,default=0.);p.add_argument('--register',action='store_true')
    a=p.parse_args();print(json.dumps(register(a.source,a.id,a.name,a.orientation,a.inner_roof_y,a.inner_radius,a.base_y,a.register),indent=2))
if __name__=='__main__':main()
