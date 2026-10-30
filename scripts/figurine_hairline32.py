"""V3.2: lift detected hair regions onto the source scalp, then trace their boundary.

No angular hairline template, fixed sideburn/nape length or decorative edge wave.
Unknown surface labels are harmonic interpolation, explicitly reported as unknown.
"""
import json
import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra
from PIL import Image, ImageDraw
import trimesh
from im2_diagnostics import surface_depth


def project(points,camera):
    q=np.column_stack((points,np.ones(len(points))))@camera.T
    xy=q[:,:2]/q[:,2,None]
    return xy,q[:,2]


def collect_observations(mesh,cameras,masks,depths,labels):
    """Visibility-weighted observed mask labels; background/occlusion are unknown."""
    vertices=np.asarray(mesh.vertices);normals=np.asarray(mesh.vertex_normals)
    total=np.zeros(len(vertices));weight=np.zeros(len(vertices));view_records=[]
    for camera,mask,depth,label in zip(cameras,masks,depths,labels):
        small=np.asarray(Image.fromarray(np.uint8(label)).resize((mask.shape[1],mask.shape[0]),Image.Resampling.NEAREST))
        person=np.isin(small,[1,2,3,4,5,7,8,10,11,12,13,14,16])
        signed=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
        xy,z=project(vertices,camera)
        ix=np.floor(xy[:,0]).astype(int);iy=np.floor(xy[:,1]).astype(int)
        valid=(z>0)&(ix>=0)&(ix<mask.shape[1])&(iy>=0)&(iy<mask.shape[0])
        ids=np.flatnonzero(valid)
        camera_center=-np.linalg.solve(camera[:,:3],camera[:,3])
        toward=camera_center-vertices;toward/=np.maximum(np.linalg.norm(toward,axis=1,keepdims=True),1e-12)
        facing=np.einsum('ij,ij->i',normals,toward)
        observed=depth[iy[ids],ix[ids]]
        tolerance=.24*np.linalg.norm(camera[2,:3])
        seen=np.isfinite(observed)&(z[ids]<=observed+tolerance)&(facing[ids]>.12)
        ids=ids[seen]
        reliable=mask[iy[ids],ix[ids]]|person[iy[ids],ix[ids]]
        ids=ids[reliable]
        values=ndi.map_coordinates(signed,[xy[ids,1],xy[ids,0]],order=1,mode='nearest')
        probability=np.clip(.5+values/2,0,1)
        w=np.maximum(facing[ids],0)**2
        total[ids]+=probability*w;weight[ids]+=w
        interface=mask&ndi.binary_dilation(person&~mask)
        view_records.append({'interface':interface,'observed_vertices':int(len(ids)),
            'positive_observed_vertices':int((probability>.5).sum()),'camera_center':camera_center})
    known=weight>1e-6
    observed=np.divide(total,weight,out=np.zeros_like(total),where=known)
    return observed,known,weight,view_records


def propagate_unknown(mesh,values,known):
    """Dirichlet graph interpolation; observed values are never overwritten."""
    if known.sum()<20 or not (values[known]>.5).any() or not (values[known]<.5).any():
        raise ValueError('Not enough observed hair AND skin labels to recover a detected hairline')
    edges=mesh.edges_unique;length=np.linalg.norm(mesh.vertices[edges[:,0]]-mesh.vertices[edges[:,1]],axis=1)
    weights=1/np.maximum(length,1e-6)
    graph=sparse.coo_matrix((np.r_[weights,weights],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(values),len(values))).tocsr()
    laplacian=sparse.diags(np.asarray(graph.sum(1)).ravel())-graph
    unknown=np.flatnonzero(~known);fixed=np.flatnonzero(known)
    result=values.copy()
    if len(unknown):
        result[unknown]=spsolve(laplacian[unknown][:,unknown],-laplacian[unknown][:,fixed]@values[fixed])
    if not np.isfinite(result).all():raise ValueError('Unknown scalp region could not be interpolated')
    return np.clip(result,0,1)



def stabilize_labels(mesh,field,observed,known):
    # Stabilize only the numeric margin, preserving observed hair/skin decisions.
    # Exact 0.5 (no preference) is assigned to skin, never invented as hair.
    field=np.asarray(field).copy()
    delta=field-.5;near=abs(delta)<5e-4
    field[near]=.5+np.where(delta[near]>0,5e-4,-5e-4)
    positive=np.flatnonzero(field>.5)
    edges=mesh.edges_unique
    graph=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(field),len(field))).tocsr()
    count,groups=sparse.csgraph.connected_components(graph[positive][:,positive],directed=False)
    removed_unknown=0
    for group in range(count):
        ids=positive[groups==group]
        if not (known[ids]&(observed[ids]>.5)).any():
            field[ids]=0;removed_unknown+=len(ids)
    return field,removed_unknown


def clip_surface(mesh,field,known,threshold=.5):
    """Clip triangles on the measured label contour; supports arbitrary sideburn shapes."""
    source=np.asarray(mesh.vertices);source_normals=np.asarray(mesh.vertex_normals)
    vertices=[];normals=[];observed=[];faces=[];original={};crossings={}
    def vertex(i):
        if i not in original:
            original[i]=len(vertices);vertices.append(source[i]);normals.append(source_normals[i]);observed.append(bool(known[i]))
        return original[i]
    def crossing(a,b):
        if abs(field[a]-threshold)<1e-10:return vertex(a)
        if abs(field[b]-threshold)<1e-10:return vertex(b)
        key=tuple(sorted((int(a),int(b))))
        if key not in crossings:
            t=(threshold-field[a])/(field[b]-field[a]);n=source_normals[a]*(1-t)+source_normals[b]*t;n/=max(np.linalg.norm(n),1e-12)
            crossings[key]=len(vertices);vertices.append(source[a]*(1-t)+source[b]*t);normals.append(n);observed.append(bool(known[a] and known[b]))
        return crossings[key]
    for tri in mesh.faces:
        polygon=[]
        for a,b in zip(tri,np.roll(tri,-1)):
            ina=field[a]>=threshold;inb=field[b]>=threshold
            if ina:polygon.append(vertex(int(a)))
            if ina!=inb:polygon.append(crossing(int(a),int(b)))
        polygon=list(dict.fromkeys(polygon))
        for j in range(1,len(polygon)-1):
            f=[polygon[0],polygon[j],polygon[j+1]]
            p=np.asarray([vertices[k] for k in f])
            if np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))>1e-10:faces.append(f)
    if not faces:raise ValueError('Detected hair surface is empty')
    patch=trimesh.Trimesh(vertices,faces,process=False)
    directed=np.vstack((patch.faces[:,[0,1]],patch.faces[:,[1,2]],patch.faces[:,[2,0]]))
    _,inverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    if (counts>2).any():raise ValueError('Detected hair patch has nonmanifold edges')
    boundary=directed[counts[inverse]==1]
    if not len(boundary):raise ValueError('No measured hairline boundary')
    return patch,np.asarray(normals),np.asarray(observed),boundary


def thicken_patch(patch,normals,boundary,config):
    """Attached volume with a tapered edge; never edits the source face mesh."""
    vertices=np.asarray(patch.vertices);edges=patch.edges_unique
    lengths=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
    graph=sparse.coo_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    distance=dijkstra(graph,directed=False,indices=np.unique(boundary),min_only=True)
    t=np.clip(distance/config.get('edge_taper_mm',.8),0,1);t=t*t*(3-2*t)
    thickness=.10+(config['base_mm']-.10)*t
    # A constant deep inset self-intersects on thin ear folds. Limit the inset
    # to local surface spacing while keeping the detected outer contour intact.
    counts=np.bincount(edges.ravel(),minlength=len(vertices))
    spacing=np.bincount(edges.ravel(),weights=np.repeat(lengths,2),minlength=len(vertices))/np.maximum(counts,1)
    inset=np.maximum(.005,np.minimum(config['anchor_mm'],spacing*.15))
    outer=vertices+normals*thickness[:,None];inner=vertices-normals*inset[:,None]
    n=len(vertices);faces=np.vstack((patch.faces,patch.faces[:,::-1]+n))
    sides=[]
    for a,b in boundary:sides.extend([[b,a,a+n],[b,a+n,b+n]])
    result=trimesh.Trimesh(np.vstack((outer,inner)),np.vstack((faces,sides)),process=False);result.fix_normals()
    if not result.is_volume:raise ValueError('Detected hair patch could not be made a closed positive volume')
    result.metadata['actual_inset_mm']=inset
    return result,outer,thickness


def diagnostic_overlays(output,images,cameras,masks,depths,records,patch,normals,known,boundary,outer):
    """Check the actual projected outline against detected hair/skin interfaces."""
    rows=[]
    for name,image,camera,mask,depth,record in zip(['front','left','right'],images,cameras,masks,depths,records):
        size=(mask.shape[1],mask.shape[0]);canvas=image.resize(size).convert('RGB')
        rgb=np.asarray(canvas).copy();rgb[mask]=(rgb[mask]*.45+np.array([30,210,70])*.55).astype(np.uint8);canvas=Image.fromarray(rgb)
        target=record['interface'];target_distance=ndi.distance_transform_edt(~target)
        target_y,target_x=np.nonzero(target)
        draw=ImageDraw.Draw(canvas)
        for x,y in zip(target_x,target_y):draw.point((int(x),int(y)),fill=(40,225,255))
        # Foot of the thickened edge is the visual hairline, not a fabricated cap curve.
        xy,z=project(outer,camera);view=record['camera_center']-outer;view/=np.maximum(np.linalg.norm(view,axis=1,keepdims=True),1e-12)
        facing=np.einsum('ij,ij->i',normals,view)
        ix=np.floor(xy[:,0]).astype(int);iy=np.floor(xy[:,1]).astype(int)
        valid=(z>0)&(ix>=0)&(iy>=0)&(ix<size[0])&(iy<size[1])&(facing>.1)
        ids=np.flatnonzero(valid);limit=.40*np.linalg.norm(camera[2,:3]);valid[ids]&=np.isfinite(depth[iy[ids],ix[ids]])&(z[ids]<=depth[iy[ids],ix[ids]]+limit)
        raster=Image.new('L',size,0);rd=ImageDraw.Draw(raster)
        distances=[];visible_edges=0
        for a,b in boundary:
            if not(valid[a] and valid[b]):continue
            line=[tuple(xy[a]),tuple(xy[b])];color=(255,85,85) if known[a] and known[b] else (255,180,40)
            draw.line(line,fill=color,width=1);rd.line(line,fill=255,width=1);visible_edges+=1
            mid=(xy[a]+xy[b])/2
            distances.append(float(ndi.map_coordinates(target_distance,[[mid[1]],[mid[0]]],order=1,mode='nearest')[0]))
        prediction=np.asarray(raster)>0
        if not prediction.any() or not len(distances):raise ValueError('No visible projected hairline in '+name)
        reverse=ndi.distance_transform_edt(~prediction)[target]
        row={'view':name,'detected_interface_pixels':int(target.sum()),'visible_projected_edges':visible_edges,
            'median_projected_to_detected_px':float(np.median(distances)),'p95_projected_to_detected_px':float(np.quantile(distances,.95)),
            'detected_interface_within_3px_fraction':float((reverse<=3).mean()) if len(reverse) else None,
            'scope':'512px-wide consistency with detected masks and recovered cameras; not independent ground truth'}
        canvas.save(output/(name+'.hairline_fit.png'));rows.append(row)
    return rows


def refine_source_surface(head,source_face_count=None):
    """Track artificial mouth/neck closure faces through exact linear subdivision."""
    excluded=np.zeros(len(head.faces),bool)
    if source_face_count is not None:excluded[source_face_count:]=True
    work=head.copy()
    for _ in range(2):
        vertices,faces=trimesh.remesh.subdivide(work.vertices,work.faces)
        # Full subdivision emits four consecutive children per original face.
        excluded=np.repeat(excluded,4)
        work=trimesh.Trimesh(vertices,faces,process=False)
    excluded_vertices=np.unique(work.faces[excluded])
    return work,excluded_vertices


def detected_hair(head,cameras,masks,depths,labels,images,output,config,source_face_count=None):
    # Refinement captures narrow temples/sideburns without a hand-designed polar cap.
    work,closure_vertices=refine_source_surface(head,source_face_count)
    observed,known,weight,records=collect_observations(work,cameras,masks,depths,labels)
    # Synthetic closure geometry is not observed scalp and must never sprout hair.
    fixed=known.copy();values=observed.copy()
    fixed[closure_vertices]=True;values[closure_vertices]=0
    field=propagate_unknown(work,values,fixed)
    field[closure_vertices]=0
    field,removed_unknown=stabilize_labels(work,field,observed,known)
    patch,normals,patch_known,boundary=clip_surface(work,field,known)
    hair,outer,thickness=thicken_patch(patch,normals,boundary,config)
    # Build a single exterior from the detected hair patch and its skin complement.
    # This avoids intersecting an inward-offset ribbon with thin ear folds.
    skin,skin_normals,_,skin_boundary=clip_surface(work,1-field,known)
    skin_points=np.asarray(skin.vertices).copy()
    skin_ids=np.unique(skin_boundary)
    skin_points[skin_ids]+=skin_normals[skin_ids]*.10
    triangles=np.vstack((outer[patch.faces],skin_points[skin.faces]))
    points,inverse=np.unique(np.round(triangles.reshape(-1,3),8),axis=0,return_inverse=True)
    print_head=trimesh.Trimesh(points,inverse.reshape(-1,3),process=False)
    print_head.fix_normals()
    if not print_head.is_volume:raise ValueError('Detected exterior and skin complement do not form a closed source head')
    hair.metadata['print_head_mesh']=print_head
    hair.metadata['observed_support']=np.r_[patch_known,patch_known]
    fits=diagnostic_overlays(output,images,cameras,masks,depths,records,patch,normals,patch_known,boundary,outer)
    edge_lengths=np.linalg.norm(patch.vertices[boundary[:,0]]-patch.vertices[boundary[:,1]],axis=1)
    unknown_edges=~(patch_known[boundary[:,0]]&patch_known[boundary[:,1]])
    samples={'unit':'mm in source-head +Y-up coordinates','points_mm':outer.tolist(),'point_has_observed_support':patch_known.tolist(),
        'boundary_edges':boundary.tolist(),'unknown_boundary_edges':unknown_edges.tolist(),
        'unknown_policy':'harmonic interpolation between observed labels, not a detected back contour'}
    (output/'hairline.detected.json').write_text(json.dumps(samples,separators=(',',':')),encoding='utf-8')
    return hair,{'method':'source-surface mask labels and extracted iso-contour; no angular cap template',
        'independent_strands':False,'fixed_hairline_prior':False,'decorative_edge_waves':False,'observed_values_overwritten_by_smoothing':False,'classification_margin_tolerance':.0005,
        'synthetic_closure_vertices_excluded_from_hair':len(closure_vertices),'working_vertices':len(work.vertices),'observed_vertices':int(known.sum()),'unknown_vertices':int((~known).sum()),
        'unsupported_hair_vertices_removed':removed_unknown,'boundary_edges':len(boundary),'unknown_boundary_length_fraction':float(edge_lengths[unknown_edges].sum()/max(edge_lengths.sum(),1e-12)),
        'hidden_regions':'harmonic interpolation between observed labels; explicitly not detected',
        'minimum_constructed_offset_sum_mm':float((thickness+hair.metadata['actual_inset_mm']).min()),
        'boundary_fits':fits,'parameters':config,'human_reviewed':False,'requires_visual_review':True,
        'surface_detail':'disabled to review the detected hairline first','print_surface':'detected outer hair patch stitched to original skin complement; .10mm seam transition; no scalp voxelization'}
