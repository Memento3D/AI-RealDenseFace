"""Read-only mesh audit. No reconstruction, repair, smoothing or identity score."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from .common import sha, write


def load_mesh(path):
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        # dump applies each node transform, including repeated geometry instances.
        parts = list(loaded.dump())
        if not parts:
            raise ValueError('Empty mesh scene')
        loaded = trimesh.util.concatenate(parts)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError('Expected triangle mesh')
    if not len(loaded.faces) or not np.isfinite(loaded.vertices).all():
        raise ValueError('Empty or nonfinite mesh')
    return loaded


def metrics(mesh):
    faces = np.asarray(mesh.faces)
    area = mesh.area_faces
    components = mesh.split(only_watertight=False)
    return {
        'vertices': int(len(mesh.vertices)), 'triangles': int(len(faces)),
        'watertight': bool(mesh.is_watertight),
        'winding_consistent': bool(mesh.is_winding_consistent),
        'positive_volume': bool(mesh.is_volume),
        'signed_volume_native': float(mesh.volume),
        'extents_native': mesh.extents.tolist(),
        'connected_components': len(components),
        'zero_area_triangles': int((area <= 1e-16).sum()),
        'boundary_edges': int((np.bincount(mesh.edges_unique_inverse) == 1).sum()),
        'nonmanifold_edges': int((np.bincount(mesh.edges_unique_inverse) > 2).sum()),
        'self_intersections_verified': False, 'minimum_thickness_verified': False,
        'identity_verified': False, 'ready_for_machine': False,
        'units': 'native; not calibrated millimeters',
    }


def render(mesh, path, gray=False, size=320):
    """Orthographic z-buffer, per-face diffuse shading; no face subsampling.

    Native TripoSR axes: Z up; first view along +X. Cameras are NOT fitted
    to input images. Smoothness/likeness cannot be scored from this alone.
    """
    v = np.asarray(mesh.vertices, dtype=float).copy()
    v -= (v.min(0) + v.max(0)) / 2
    scale = .84 * size / max(np.ptp(v, axis=0).max(), 1e-12)
    f = np.asarray(mesh.faces)
    if not gray and mesh.visual.kind == 'vertex':
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)
    else:
        colors = np.full((len(v), 3), 185.)
    face_colors = colors[f].mean(1)
    sheet = Image.new('RGB', (size*4, size+24), (26,31,38))
    draw = ImageDraw.Draw(sheet)
    for index, deg in enumerate((0,90,180,270)):
        a = np.deg2rad(deg)
        depth = v[:,0]*np.cos(a) + v[:,1]*np.sin(a)
        horizontal = -v[:,0]*np.sin(a) + v[:,1]*np.cos(a)
        xy = np.column_stack((horizontal*scale+size*.5, -v[:,2]*scale+size*.5))
        # True nearest surface visibility, interpolated depth; no painter gaps.
        pixels = np.full((size,size,3), (35,40,47), dtype=np.uint8)
        zbuffer = np.full((size,size), -np.inf)
        normals = np.asarray(mesh.face_normals)
        view_n = np.column_stack((-normals[:,0]*np.sin(a)+normals[:,1]*np.cos(a),
                                  normals[:,2], normals[:,0]*np.cos(a)+normals[:,1]*np.sin(a)))
        light = np.array([-.35,.5,.8]); light /= np.linalg.norm(light)
        intensity = .35 + .65*np.maximum(view_n@light, 0)
        for fi, face in enumerate(f):
            t = xy[face]
            lo = np.maximum(np.floor(t.min(0)).astype(int), 0)
            hi = np.minimum(np.ceil(t.max(0)).astype(int), size-1)
            if np.any(lo > hi): continue
            x0,y0=t[0];x1,y1=t[1];x2,y2=t[2]
            den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
            if abs(den)<1e-10: continue
            yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
            w0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den
            w1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den
            w2=1-w0-w1
            z=w0*depth[face[0]]+w1*depth[face[1]]+w2*depth[face[2]]
            zb=zbuffer[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
            mask=(w0>=-1e-8)&(w1>=-1e-8)&(w2>=-1e-8)&(z>zb)
            zb[mask]=z[mask]
            rgb=np.clip(face_colors[fi]*(intensity[fi] if gray else 1.),0,255).astype(np.uint8)
            pixels[lo[1]:hi[1]+1,lo[0]:hi[0]+1][mask]=rgb
        sheet.paste(Image.fromarray(pixels), (index*size,24))
        draw.text((index*size+8,5), str(deg)+' deg / native', fill='white')
    sheet.save(path)


def audit(mesh_path, output, source_report=None):
    mesh_path=Path(mesh_path).resolve();output=Path(output).resolve()
    # Verify provenance BEFORE creating output, and never edit source bytes.
    digest=sha(mesh_path)
    provenance={'report_supplied':source_report is not None, 'mesh_hash_verified':False}
    if source_report:
        r=json.loads(Path(source_report).read_text(encoding='utf-8-sig'))
        expected=r.get('output_sha256',{}).get(mesh_path.name)
        if not expected or expected != digest:
            raise ValueError('SOURCE_HASH_MISMATCH: mesh does not match supplied report')
        provenance.update(mesh_hash_verified=True,source_report_sha256=sha(source_report),
                          source_status=r.get('status'),source_model_revision=r.get('model_revision'))
    mesh=load_mesh(mesh_path)
    result={'status':'MESH_AUDITED','mesh_sha256':digest,'geometry':metrics(mesh),
            'provenance':provenance,'geometry_modified':False,'hair_isolated':False,
            'face_identity_verified':False,'ready_for_machine':False,
            'camera_registration_verified':False,
            'purpose':'separate visible geometry from color; no automatic likeness verdict'}
    output.mkdir(parents=True,exist_ok=False)
    render(mesh,output/'color_unlit.png')
    render(mesh,output/'geometry_gray.png',gray=True)
    result['source_bytes_unchanged']=sha(mesh_path)==digest
    write(output/'audit.json',result)
    (output/'review.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>TripoSR mesh audit</title>'
        '<style>body{background:#171c24;color:white;font:16px system-ui;margin:24px}img{max-width:100%}a{color:#acf}</style>'
        '<h1>TripoSR: color / geometry</h1><p>Same mesh, same cameras. Native units; no repair. '
        'Gray uses face-normal diffuse shading; unlit uses vertex colors. Not a physical-print simulation.</p>'
        '<h2>Color without lighting</h2><img src="color_unlit.png">'
        '<h2>Gray with lighting</h2><img src="geometry_gray.png">'
        '<p>Watertight is not a likeness or manufacturing verdict.</p><a href="audit.json">audit.json</a>'
        '<pre>'+html.escape(json.dumps(result,indent=2))+'</pre>', encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path)
    a=p.parse_args();audit(a.mesh,a.output,a.report)
    print('Audit:',a.output/'review.html')

if __name__=='__main__':main()
