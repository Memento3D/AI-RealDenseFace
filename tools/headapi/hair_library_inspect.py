"""Blender-only, read-only inspection of locally obtained hair asset libraries."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def inspect(asset, output):
    import bpy
    import bmesh
    from mathutils import Vector

    asset = Path(asset).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a fresh inspection directory')
    if asset.suffix.lower() != '.blend':
        raise ValueError('This inspector currently accepts .blend libraries only')
    before = hashlib.sha256(asset.read_bytes()).hexdigest()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(asset), link=False) as (source, loaded):
        loaded.objects = list(source.objects)
    rows = []
    for obj in loaded.objects:
        if obj is None:
            continue
        row = {'name': obj.name, 'type': obj.type,
               'modifiers': [{'name': m.name, 'type': m.type} for m in obj.modifiers],
               'candidate_name_hint': any(s in obj.name.lower() for s in ('hair', 'hairstyle', 'groom'))}
        if obj.type == 'MESH':
            mesh = obj.data
            bm = bmesh.new()
            bm.from_mesh(mesh)
            row.update(vertices=len(mesh.vertices), polygons=len(mesh.polygons),
                       boundary_edges=sum(e.is_boundary for e in bm.edges),
                       nonmanifold_edges=sum(not e.is_manifold for e in bm.edges),
                       loose_vertices=sum(not v.link_faces for v in bm.verts))
            bm.free()
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            row['world_bounds'] = [[min(c[k] for c in corners) for k in range(3)],
                                   [max(c[k] for c in corners) for k in range(3)]]
            materials = []
            for material in mesh.materials:
                if material is None:
                    continue
                nodes = list(material.node_tree.nodes) if material.use_nodes and material.node_tree else []
                images = [n.image.name for n in nodes if n.type == 'TEX_IMAGE' and n.image]
                materials.append({'name': material.name, 'image_names': images,
                                  'alpha_links': sum(link.to_socket.name == 'Alpha' for link in material.node_tree.links)
                                  if material.use_nodes and material.node_tree else 0})
            row['materials'] = materials
            row['mesh_closed_topologically'] = row['nonmanifold_edges'] == 0 and row['boundary_edges'] == 0
        rows.append(row)
    if hashlib.sha256(asset.read_bytes()).hexdigest() != before:
        raise ValueError('Source asset changed during inspection')
    output.mkdir(parents=True)
    report = {'status': 'LIBRARY_INSPECTED_NOT_FITTED', 'source_path': str(asset),
              'source_sha256': before, 'blender_version': bpy.app.version_string,
              'objects': rows, 'hair_mesh_candidates': [r['name'] for r in rows
                  if r['type'] == 'MESH' and r['candidate_name_hint']],
              'source_modified': False, 'license_verified': False, 'ready_for_machine': False,
              'limitations': ['Object-name hints are not verified hairstyle labels',
                              'Counts describe base mesh, not evaluated modifiers',
                              'Closed topology does not establish printability',
                              'Licensing requires the asset-specific license file']}
    (output / 'inspection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('LIBRARY_INSPECTED_NOT_FITTED', len(rows), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    inspect(args.asset, args.output)
