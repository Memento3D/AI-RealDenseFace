"""Original figurine geometry: closed mesh candidates, image-constrained hair mass, stylized eyes/teeth."""
import numpy as np
from scipy import ndimage as ndi
import trimesh


def weld(triangles):
    points, inverse = np.unique(np.asarray(triangles).reshape(-1, 3), axis=0, return_inverse=True)
    return trimesh.Trimesh(points, inverse.reshape(-1, 3), process=False)


def close_boundaries(mesh):
    """Fan-cap existing boundary loops; intended for FLAME neck/inner mouth, not general repair."""
    vertices = mesh.vertices.tolist(); faces = mesh.faces.tolist()
    directed = np.vstack((mesh.faces[:, [0,1]], mesh.faces[:, [1,2]], mesh.faces[:, [2,0]]))
    sorted_edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError('Source head has nonmanifold edges')
    boundary = directed[counts[inverse] == 1]
    nxt = {}
    for a,b in boundary:
        if int(a) in nxt: raise ValueError('Ambiguous boundary loop')
        nxt[int(a)] = int(b)
    caps = []
    while nxt:
        first = next(iter(nxt)); current = first; loop = []
        while True:
            if current not in nxt: raise ValueError('Open boundary chain')
            loop.append(current); current = nxt.pop(current)
            if current == first: break
            if len(loop) > len(boundary): raise ValueError('Invalid boundary cycle')
        center = np.asarray(vertices)[loop].mean(0)
        index = len(vertices); vertices.append(center.tolist())
        for a,b in zip(loop, loop[1:]+loop[:1]): faces.append([index, b, a])
        caps.append({'vertices': len(loop), 'center_mm': center.tolist()})
    closed = trimesh.Trimesh(vertices, faces, process=False)
    closed.fix_normals()
    if not closed.is_watertight: raise ValueError('Boundary closure did not yield a watertight head')
    return closed, caps


def recover_projector(vertices, coordinates, image_size, upper, max_error_px=5):
    """Recover a projective map from stored upper-head correspondences. Other expressions are approximate."""
    q = np.asarray(coordinates).reshape(-1, 3)
    x = np.column_stack((vertices, np.ones(len(vertices))))
    if np.linalg.matrix_rank(x[upper]) < 4: raise ValueError('Insufficient 3D support for camera recovery')
    matrix = np.linalg.lstsq(x[upper], q[upper], rcond=None)[0].T
    pred = x @ matrix.T
    if (pred[upper, 2] <= 0).any(): raise ValueError('Invalid recovered camera depth')
    delta = pred[upper, :2]/pred[upper, 2,None]-q[upper, :2]/q[upper, 2,None]
    delta *= np.asarray(image_size)
    errors = np.linalg.norm(delta, axis=1)
    if np.quantile(errors, .95) > max_error_px:
        raise ValueError('Upper-head camera reprojection exceeds configured tolerance')
    pixels = matrix.copy()
    pixels[0] *= image_size[0]
    pixels[1] = (matrix[2]-matrix[1])*image_size[1]
    return pixels, {'median_px': float(np.median(errors)), 'p95_px': float(np.quantile(errors,.95)),
                    'max_px': float(errors.max()), 'scope': 'Upper-head fit to stored projections, not independent calibration'}


def grid_for(mesh, pitch, padding):
    low = np.floor((mesh.bounds[0]-padding)/pitch)*pitch
    high = np.ceil((mesh.bounds[1]+padding)/pitch)*pitch
    shape = np.ceil((high-low)/pitch).astype(int)+1
    if np.prod(shape) > 24_000_000: raise ValueError('Voxel budget exceeded; increase voxel_mm')
    return low, tuple(shape)


def voxel_solid(mesh, origin, shape, pitch):
    vox = mesh.voxelized(pitch).fill()
    ijk = np.rint((vox.points-origin)/pitch).astype(int)
    if ((ijk < 0).any() or (ijk >= np.asarray(shape)).any()): raise ValueError('Voxel bounds exceeded')
    solid = np.zeros(shape, bool); solid[tuple(ijk.T)] = True
    return solid


def mesh_from_volume(volume, origin, pitch):
    from skimage.measure import marching_cubes
    if not volume.any(): raise ValueError('Empty solid')
    field = ndi.gaussian_filter(volume.astype(np.float32), .55)
    vertices, faces, _, _ = marching_cubes(field, .5, spacing=(pitch,)*3, allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices+origin, faces, process=False)
    mesh.fix_normals()
    before=mesh.vertices.copy();volume=abs(mesh.volume)
    # Remove grid terraces before judging the actual printable surface.
    trimesh.smoothing.filter_taubin(mesh,lamb=.5,nu=.53,iterations=30)
    ratio=abs(mesh.volume)/max(volume,1e-9)
    if not .97<ratio<1.03:raise ValueError('Surface smoothing changed volume excessively')
    mesh.metadata['surface_processing']={'method':'30 alternating Taubin steps after voxel extraction','volume_ratio':float(ratio),'max_vertex_displacement_mm':float(np.linalg.norm(mesh.vertices-before,axis=1).max())}
    return mesh


def make_hair(head_solid, origin, pitch, eye_y, head_bounds, cameras, masks, head_depths, config):
    """Visual support carving plus original macro-ridges; hidden shape remains a prior."""
    shape = head_solid.shape
    outside = ndi.distance_transform_edt(~head_solid).astype(np.float32)*pitch
    inside = ndi.distance_transform_edt(head_solid).astype(np.float32)*pitch
    signed = outside-inside
    low, high = head_bounds; height = high[1]-low[1]
    center = (low+high)/2
    # Parametric crown is a design guide; silhouette limits come from the photographs.
    hair = np.zeros(shape, bool)
    clumps = config['clumps']; part = config['part']
    flat = hair.reshape(-1)
    flat_signed = signed.reshape(-1)
    for start in range(0, hair.size, 180000):
        ids = np.arange(start, min(start+180000, hair.size))
        ijk = np.column_stack(np.unravel_index(ids, shape))
        points = origin+ijk*pitch
        front = points[:,2] > center[2]+.12*(high[2]-low[2])
        floor = np.where(front, eye_y+.055*height, eye_y-.26*height)
        candidate = (points[:,1] >= floor) & (flat_signed[ids] >= -config['anchor_mm'])
        angle = np.arctan2(points[:,0]-center[0]-part*(high[0]-low[0])*.25,
                           points[:,2]-center[2])
        # Broad curved ridges form attached locks; not thousands of strands.
        phase = angle*clumps + config['sweep']*(points[:,1]-eye_y)/height*5
        ridge = np.maximum(np.cos(phase), 0)**3
        thickness = config['base_mm']+config['ridge_mm']*ridge
        candidate &= flat_signed[ids] <= thickness
        for cam, mask, depth in zip(cameras, masks, head_depths):
            q = np.column_stack((points, np.ones(len(points)))) @ cam.T
            positive = q[:,2] > 1e-8
            xy = np.zeros((len(points),2),int)
            xy[positive] = np.floor(q[positive,:2]/q[positive,2,None]).astype(int)
            in_frame = positive & (xy[:,0] >= 0) & (xy[:,0] < mask.shape[1]) & (xy[:,1] >= 0) & (xy[:,1] < mask.shape[0])
            selected = np.flatnonzero(candidate & in_frame)
            observed = depth[xy[selected,1],xy[selected,0]]
            tolerance = pitch*1.8*np.linalg.norm(cam[2,:3])
            visible = q[selected,2] <= observed+tolerance
            keep = mask[xy[selected,1],xy[selected,0]]
            candidate[selected[visible & ~keep]] = False
            # A part cannot extend beyond all valid source frames unnoticed.
            candidate[~in_frame] = False
        flat[ids] = candidate
    # Include a thin intersection with the head so the union has a load-bearing attachment.
    if hair.sum() < 100: raise ValueError('Image-constrained hair mass is empty or too small')
    return hair, {'method': 'three-view visible-surface mask constraints with parametric attached macro-ridges',
                  'hair_voxels': int(hair.sum()), 'hidden_regions': 'bounded prior thickness, not observed reconstruction',
                  'clumps': clumps, 'independent_strands': False, 'parameters': config}


def eye_mesh(center, radii, config):
    mesh = trimesh.creation.uv_sphere(radius=1, count=[24,40])
    unit = mesh.vertices.copy()
    mesh.vertices = center+unit*radii
    yaw, pitch = np.deg2rad([config['gaze_yaw_deg'], config['gaze_pitch_deg']])
    gaze = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    right = np.cross([0,1,0],gaze); right /= np.linalg.norm(right)
    up = np.cross(gaze,right)
    xx,yy,zz = unit@right,unit@up,unit@gaze
    rr = np.hypot(xx,yy)
    colors = np.tile(config['sclera_rgb'],(len(unit),1)).astype(float)
    iris = (zz > 0) & (rr < config['iris_ratio'])
    pupil = (zz > 0) & (rr < config['iris_ratio']*config['pupil_ratio'])
    colors[iris] = config['iris_rgb']; colors[pupil] = [9,8,8]
    highlight = iris & ((xx+.13)**2+(yy-.15)**2 < config['highlight_ratio']**2)
    colors[highlight] = [245,242,230]
    return mesh, np.column_stack((colors/255,np.ones(len(colors))))


def mouth_anchor(vertices, q, labels, head_corners, front_weight, scale, center):
    uv = q[:,:2]/q[:,2,None]
    xy = np.column_stack((uv[:,0]*labels.shape[1], (1-uv[:,1])*labels.shape[0])).astype(int)
    valid = (xy[:,0]>=0)&(xy[:,0]<labels.shape[1])&(xy[:,1]>=0)&(xy[:,1]<labels.shape[0])&head_corners&(front_weight>1e-8)
    semantic = np.zeros(len(vertices),bool)
    semantic[valid] = np.isin(labels[xy[valid,1],xy[valid,0]], [12,13])
    points = (vertices[semantic]-center)*scale
    if len(points)<12: raise ValueError('Not enough visible lip vertices to place an individualized mouth')
    lo,hi = np.quantile(points,[.03,.97],axis=0)
    anchor = np.median(points,axis=0)
    width = hi[0]-lo[0]
    if width <= .5: raise ValueError('Invalid lip width')
    return anchor,width,{'method':'visible FLAME lip vertices selected using front semantic lip pixels',
                        'support_corners':int(semantic.sum()),'center_mm':anchor.tolist(),'width_mm':float(width)}


def smile_deform(vertices, anchor, width, config):
    delta = vertices-anchor
    x,y,z = delta.T
    front = np.exp(-((z/(width*.35))**4))
    horizontal = np.exp(-((x/(width*.58))**6))
    vertical = np.exp(-((y/(width*.22))**4))
    weight = front*horizontal*vertical
    # A declared expression preset, not recovered smile anatomy.
    opening = np.tanh(y/max(width*.025,.02))*config['gap_mm']*.5
    corners = config['corner_lift_mm']*np.minimum((x/(width*.5))**2,1)
    out = vertices.copy(); out[:,1] += weight*(opening+corners)
    return out, {'expression_origin':'synthetic smile feasibility preset',
                 'max_displacement_mm':float(np.linalg.norm(out-vertices,axis=1).max()),
                 'modified_vertices':int((np.linalg.norm(out-vertices,axis=1)>1e-4).sum())}


def teeth_band(anchor, width, config, surface_z=None):
    # Continuous shallow curved relief. Dividers are colors only, never separated teeth.
    w=width*.78; h=config['gap_mm']*.72; d=max(.45,config['gap_mm']*.65)
    xs=np.linspace(-w/2,w/2,25)
    verts=[]
    for x in xs:
        z=anchor[2]-.25-.35*(x/(w/2))**2
        curve=config.get('corner_lift_mm',0)*(x/(w/2))**2
        taper=max(.12,(1-(x/(w/2))**2)**.45)
        ring=[]
        for dy,front in [(-h/2,False),(h/2,False),(h/2,True),(-h/2,True)]:
            y=anchor[1]+curve+dy*taper
            zz=(surface_z(anchor[0]+x,y)+.055 if surface_z else z+d/2)
            ring.append([anchor[0]+x,y,zz if front else zz-d])
        verts.extend(ring)
    faces=[]
    for i in range(len(xs)-1):
        for j in range(4):
            a=i*4+j;b=i*4+(j+1)%4;c=(i+1)*4+(j+1)%4;dd=(i+1)*4+j
            faces.extend([[a,b,c],[a,c,dd]])
    faces.extend([[0,2,1],[0,3,2]])
    a=(len(xs)-1)*4;faces.extend([[a,a+1,a+2],[a,a+2,a+3]])
    mesh=trimesh.Trimesh(verts,faces,process=False);mesh.fix_normals()
    colors=np.tile([.88,.86,.77,1.],(len(verts),1))
    # Subtle dividers at selected band samples, for the color guide only.
    for i in (6,10,14,18): colors[i*4:(i+1)*4,:3]=[.48,.43,.38]
    return mesh,colors


def solid_report(mesh):
    pieces=mesh.split(only_watertight=False)
    return {'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),
            'connected_components':len(pieces),'volume_mm3':float(abs(mesh.volume)),
            'bounds_mm':mesh.bounds.tolist(),'triangles':len(mesh.faces),
            'surface_processing':mesh.metadata.get('surface_processing',{}),
            'slicer_verified':False,'physical_print_verified':False,
            'minimum_wall_thickness_verified':False,'support_strategy_verified':False}


def union_print_parts(parts, allow_tiny_surface_slivers=False):
    """Exact surface boolean keeps the source face free of voxel terraces.

    Manifold returns oriented cavity shells. Filling their enclosed voids is
    deliberate for this solid prototype. Detached positive-volume pieces fail.
    """
    if not all(p.is_volume for p in parts):raise ValueError('Every print part must be a closed positive volume')
    result=trimesh.boolean.union(parts,engine='manifold',check_volume=True)
    if allow_tiny_surface_slivers:
        from manifold3d import Manifold,Mesh
        # Collapse coplanar working subdivisions without voxelizing the face.
        before_volume=float(abs(result.volume))
        manifold=Manifold(Mesh(vert_properties=np.asarray(result.vertices,dtype=np.float32),tri_verts=np.asarray(result.faces,dtype=np.uint32)))
        simplified=manifold.as_original().set_tolerance(.0001).to_mesh()
        result=trimesh.Trimesh(simplified.vert_properties[:,:3],simplified.tri_verts,process=False)
        if abs(abs(result.volume)-before_volume)>max(.01,before_volume*.0001):raise ValueError('Coplanar simplification changed volume beyond numerical tolerance')
        # Weld float32-coincident vertices and discard zero-area boolean ribbons.
        result.merge_vertices(digits_vertex=5)
        result.update_faces(result.nondegenerate_faces(height=1e-7))
        result.update_faces(result.unique_faces())
        result.remove_unreferenced_vertices()
    pieces=sorted(result.split(only_watertight=False),key=lambda m:abs(m.volume),reverse=True)
    if not pieces or pieces[0].volume<=0:raise ValueError('No positive exterior shell after union')
    outer=pieces[0];cavities=[];slivers=[]
    volume_limit=float(outer.extents.max()**3*2e-8)
    extent_limit=float(outer.extents.max()*.01)
    for piece in pieces[1:]:
        if piece.volume>=0:
            if allow_tiny_surface_slivers and (piece.volume<=1e-10 or (piece.volume<=volume_limit and piece.extents.max()<=extent_limit)):
                slivers.append({'removed_volume_mm3':float(piece.volume),'max_extent_mm':float(piece.extents.max()),'triangles':len(piece.faces)})
                continue
            raise ValueError('Detached positive-volume print part; fix its attachment: volume_mm3='+str(float(piece.volume))+'; max_extent_mm='+str(float(piece.extents.max())))
        if (piece.bounds[0]<outer.bounds[0]-1e-4).any() or (piece.bounds[1]>outer.bounds[1]+1e-4).any():raise ValueError('Invalid cavity bounds')
        cavities.append({'filled_void_mm3':float(-piece.volume),'triangles':len(piece.faces)})
    outer.metadata['surface_processing']={'method':'Manifold surface boolean; original head is not voxel-resampled','filled_enclosed_cavities':cavities,'removed_tiny_surface_slivers':slivers,'sliver_volume_limit_mm3':volume_limit if allow_tiny_surface_slivers else 0,'coplanar_simplification_tolerance_mm':.0001 if allow_tiny_surface_slivers else 0,'hair_surface':'voxel-designed and Taubin-smoothed before union'}
    return outer
