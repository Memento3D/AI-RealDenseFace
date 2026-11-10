"""Remove thin lower-neck flare from a print-only derivative, preserving source runs."""
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4
import trimesh
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/memento3d_hair_on_head/runtime')]
from figurine_geometry import close_boundaries
from figurine2_boolean import solid_union
from tools.headapi.print_study import sha,write


def clip_above(mesh,z):
    # Triangle/half-space clipping preserves winding; cap is created separately.
    triangles=[]
    for triangle in mesh.triangles:
        polygon=[]
        for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
            inside_a=a[2]>=z;inside_b=b[2]>=z
            if inside_a:polygon.append(a)
            if inside_a!=inside_b:
                point=a+(b-a)*((z-a[2])/(b[2]-a[2]));point[2]=z;polygon.append(point)
        for i in range(1,len(polygon)-1):triangles.append([polygon[0],polygon[i],polygon[i+1]])
    triangles=np.asarray(triangles)
    points,inverse=np.unique(np.round(triangles.reshape(-1,3),7),axis=0,return_inverse=True)
    return trimesh.Trimesh(points,inverse.reshape(-1,3),process=True)


def run():
    source=ROOT/'runs/experiments/input04-print-study-01'
    output=ROOT/'runs/experiments'/('input04-print-neck-'+uuid4().hex);output.mkdir(parents=True,exist_ok=False)
    print(output,flush=True)
    report=json.loads((source/'report.json').read_text());report['policy']='print-only crop below old Z=8mm; reclose neck and attach fresh pedestal'
    report['source_report_sha256']=sha(source/'report.json');report['code_sha256']=sha(Path(__file__))
    for name in ['baseline','refined']:
        folder=output/name;folder.mkdir();(folder/'rdf').mkdir();(folder/'hair').mkdir()
        shutil.copyfile(source/name/'rdf/projection_bundle.npz',folder/'rdf/projection_bundle.npz')
        shutil.copyfile(source/name/'hair/hair_cap_native.glb',folder/'hair/hair_cap_native.glb')
        path=source/name/'head_hair_pedestal_mm.stl';original=trimesh.load(path,force='mesh',process=True)
        cut=clip_above(original,8)
        cut.remove_unreferenced_vertices();cut,caps=close_boundaries(cut)
        ring=cut.vertices[cut.vertices[:,2]<8.001];center=ring[:,:2].mean(0)
        radius=float(np.linalg.norm(ring[:,:2]-center,axis=1).max()+1.5)
        base=trimesh.creation.cylinder(radius=radius,height=3,sections=96);base.apply_translation([*center,7.5])
        solid=solid_union([cut,base]);solid.apply_translation([0,0,-6])
        solid.export(folder/'head_hair_pedestal_mm.stl');solid.export(folder/'head_hair_pedestal_mm.glb')
        row=report['variants'][name];row.update(source_stl_sha256=sha(path),additional_export_z_shift_mm=6,
            print_only_cut_old_z_mm=8,cut_caps=caps,stl_sha256=sha(folder/'head_hair_pedestal_mm.stl'),
            geometry={'watertight':bool(solid.is_watertight),'positive_volume':bool(solid.is_volume),
                'components':len(solid.split()),'winding_consistent':bool(solid.is_winding_consistent),
                'bounds_mm':solid.bounds.tolist(),'volume_mm3':float(solid.volume),'self_intersections_verified':False})
        row.pop('thickness_proxy',None);row.pop('source_head_distance_to_union_mm',None)
        print(name+' neck derivative exported',flush=True)
    write(output/'report.json',report)


if __name__=='__main__':run()
