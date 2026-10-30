"""Manufacturing fallback for hair ONLY. Never resamples face/eyes/skin.

Bounded voxel closing resolves sub-grid pinches in a supplied/deformed style.
This is logged geometric simplification, not a minimum-wall guarantee.
"""
import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

def repair_hair(mesh,pitch=.10):
    if not .08<=pitch<=.16:raise ValueError('Hair repair pitch outside declared range')
    vox=mesh.voxelized(pitch).fill()
    if np.prod(vox.shape)>16000000:raise ValueError('Hair-only repair voxel budget exceeded')
    material=np.pad(vox.matrix,2)
    closed=ndi.binary_closing(material,structure=ndi.generate_binary_structure(3,1),iterations=1)
    v,f,_,_=marching_cubes(closed.astype(np.float32),.5,spacing=(pitch,pitch,pitch),allow_degenerate=False)
    v+=vox.transform[:3,3]-2*pitch
    repaired=trimesh.Trimesh(v,f,process=True);repaired.fix_normals()
    trimesh.smoothing.filter_taubin(repaired,lamb=.45,nu=.46,iterations=2)
    if not repaired.is_volume or len(repaired.split())!=1:raise ValueError('Hair repair did not produce one closed volume')
    delta=abs(abs(repaired.volume)-abs(mesh.volume))/max(abs(mesh.volume),1e-6)
    if delta>.08:raise ValueError('Hair repair changed volume by more than 8 percent')
    sample=repaired.vertices[::max(1,len(repaired.vertices)//300)]
    _,distance,_=trimesh.proximity.closest_point(mesh,sample)
    if distance.max()>.30:raise ValueError('Hair repair surface sample error exceeds .30 mm')
    return repaired,{'method':'hair-only voxel material fill, one-step 6-neighbor closing and two Taubin iterations',
        'voxel_pitch_mm':pitch,'source_face_resampled':False,'relative_volume_change':float(delta),
        'sampled_surface_error_max_mm':float(distance.max()),'sample_count':len(sample),
        'exact_hausdorff_verified':False,'minimum_wall_verified':False,
        'detail_policy':'sub-grid hair features may merge; broad supplied style retained approximately'}
