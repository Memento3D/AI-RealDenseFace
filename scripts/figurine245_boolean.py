"""Double precision union plus STL roundtrip guard; never voxelize source face."""
import io
import numpy as np
import trimesh
from manifold3d import Manifold,Mesh64,Error

def solid_union(parts):
    items=[]
    for part in parts:
        if not part.is_volume:raise ValueError('Union input is not a closed positive volume')
        item=Manifold(Mesh64(vert_properties=np.asarray(part.vertices,dtype=np.float64),tri_verts=np.asarray(part.faces,dtype=np.uint64)))
        if item.status()!=Error.NoError:raise ValueError('Manifold rejected an input part: '+str(item.status()))
        items.append(item)
    combined=items[0]
    for item in items[1:]:combined=combined+item
    if combined.status()!=Error.NoError:raise ValueError('Boolean union failed')
    outer=[];filled=[]
    for component in combined.decompose():
        if component.volume()>0:outer.append(component)
        else:filled.append(float(-component.volume()))
    tiny=[];kept=[]
    for item in outer:
        box=np.asarray(item.bounding_box());diagonal=float(np.linalg.norm(box[3:]-box[:3]))
        if item.volume()<1e-7 and diagonal<.02:tiny.append({'volume_mm3':float(item.volume()),'bbox_diagonal_mm':diagonal})
        else:kept.append(item)
    outer=kept
    if len(outer)!=1 or sum(x['volume_mm3'] for x in tiny)>1e-6:raise ValueError('Detached material after union: '+str([p.volume() for p in outer]))
    volume=float(outer[0].volume());history=[]
    for tolerance in [.0001,.001,.003,.006,.01]:
        candidate=outer[0].as_original().set_tolerance(tolerance)
        raw=candidate.to_mesh64();mesh=trimesh.Trimesh(raw.vert_properties[:,:3],raw.tri_verts,process=False)
        # STL is float32: verify the file representation, not just float64 mesh.
        reread=trimesh.load(io.BytesIO(mesh.export(file_type='stl')),file_type='stl',process=True)
        passed=bool(reread.is_volume and reread.is_winding_consistent and len(reread.split())==1)
        delta=float(abs(abs(reread.volume)-volume));history.append({'tolerance_mm':tolerance,'stl_roundtrip_pass':passed,'volume_change_mm3':delta})
        if passed and delta<=max(.02,volume*.0002):
            reread.metadata['surface_processing']={'method':'float64 Manifold surface union, enclosed void fill, bounded tolerance and float32 STL roundtrip',
                'source_face_voxelized':False,'discarded_numerical_fragments':tiny,'numerical_fragment_volume_limit_mm3':1e-7,'numerical_fragment_diagonal_limit_mm':.02,'filled_void_volumes_mm3':filled,'selected_tolerance_mm':tolerance,'trials':history}
            return reread
    raise ValueError('STL roundtrip topology failed within allowed tolerance/volume budget: '+str(history))
