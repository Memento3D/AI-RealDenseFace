"""Blender-only exploratory asset fit; does not certify anatomical fit or printing."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(objects):
    h = hashlib.sha256()
    for obj in objects:
        h.update(obj.data.name.encode())
        for v in obj.data.vertices:
            h.update(str(tuple(v.co)).encode())
        for polygon in obj.data.polygons:
            h.update(str(tuple(polygon.vertices)).encode())
        for uv in obj.data.uv_layers:
            for item in uv.data:
                h.update(str(tuple(item.uv)).encode())
    return h.hexdigest()


def main():
    import sys
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--case', default='input07', choices=['input07', 'input04', 'capture02'])
    p.add_argument('--surface-fit', action='store_true')
    p.add_argument('--cage-fit', action='store_true')
    p.add_argument('--refine-asset', action='store_true', help='Requires --cage-fit; extend rear and compare smoothing/remeshing')
    args = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.refine_asset and not args.cage_fit:
        p.error('--refine-asset requires --cage-fit')
    root = Path(__file__).resolve().parents[2]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    head_path = root / f'runs/buzz-framework-unified-01/attempt-02/{args.case}/gap-0_002/head.glb'
    asset_path = root / 'runs/male-hair-library-01/male-source-01/male-hair-source.blend'
    pins = {str(path.relative_to(root)): digest(path) for path in (head_path, asset_path)}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(head_path))
    heads = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    original = signature(heads)
    original_materials = {obj.name: list(obj.data.materials) for obj in heads}
    review_materials = {}
    for obj in heads:
        copies = []
        for mat in original_materials[obj.name]:
            copy = mat.copy()
            tree = copy.node_tree
            tex = next((n for n in tree.nodes if n.type == 'TEX_IMAGE'), None)
            output = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
            emission = tree.nodes.new('ShaderNodeEmission')
            if tex:
                tree.links.new(tex.outputs['Color'], emission.inputs['Color'])
            tree.links.new(emission.outputs[0], output.inputs['Surface'])
            copies.append(copy)
        review_materials[obj.name] = copies
    bpy.context.view_layer.update()
    coords = [obj.matrix_world @ v.co for obj in heads for v in obj.data.vertices]
    top = max(v.z for v in coords)
    upper = [v for v in coords if v.z > top - .085]
    width = max(v.x for v in upper) - min(v.x for v in upper)
    scalp_center = Vector(((max(v.x for v in upper) + min(v.x for v in upper)) / 2,
                           (max(v.y for v in upper) + min(v.y for v in upper)) / 2,
                           top - width * .43))
    vertices, polygons = [], []
    for obj in heads:
        offset = len(vertices)
        vertices.extend(obj.matrix_world @ v.co for v in obj.data.vertices)
        polygons.extend(tuple(offset + i for i in f.vertices) for f in obj.data.polygons)
    scalp_bvh = BVHTree.FromPolygons(vertices, polygons)
    with bpy.data.libraries.load(str(asset_path), link=False) as (_, data):
        data.objects = ['Male_Hair']
    hair = data.objects[0]
    bpy.context.scene.collection.objects.link(hair)
    hair.parent = None
    hair.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()
    # Input GLB is converted by Blender to Z-up; this library is already Z-up.
    source = [v.co.copy() for v in hair.data.vertices]
    lo = Vector([min(v[k] for v in source) for k in range(3)])
    hi = Vector([max(v[k] for v in source) for k in range(3)])
    center, size = (lo + hi) / 2, hi - lo
    material = bpy.data.materials.new('Neutral dark brown hair')
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (.025, .018, .014, 1)
    shader.inputs['Roughness'].default_value = .65
    hair.data.materials.clear()
    hair.data.materials.append(material)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 12
    scene.render.resolution_x = scene.render.resolution_y = 520
    scene.render.resolution_percentage = 100
    world = bpy.data.worlds.new('Review world')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.13, .15, .18, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .7
    scene.world = world
    look = Vector((0, .035, .035))
    for name, position, power in [('Key', (-.6, -.9, 1), 55), ('Fill', (.7, -.3, .6), 25)]:
        d = bpy.data.lights.new(name, 'AREA')
        d.energy, d.size = power, 1
        o = bpy.data.objects.new(name, d)
        scene.collection.objects.link(o)
        o.location = Vector(position)
        o.rotation_euler = (look - o.location).to_track_quat('-Z', 'Y').to_euler()
    camera = bpy.data.objects.new('Review camera', bpy.data.cameras.new('Review camera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = .42
    presets = [('base', None, None),
               ('original-volume', (.205, .220, .170), (0, .045, .100)),
               ('compact', (.195, .205, .140), (0, .035, .078))]
    if args.surface_fit:
        presets = [('base', None, None), ('surface-low', (.195, .205, .140), (0, .035, .078)),
                   ('surface-volume', (.195, .205, .140), (0, .035, .078))]
    if args.cage_fit:
        presets = [('base', None, None), ('cage-close', (.195, .205, .140), (0, .035, .078)),
                   ('cage-lift', (.195, .205, .140), (0, .035, .078))]
    report = {'status': 'EXPLORATORY_AFFINE_FIT_NOT_CALIBRATED', 'inputs': pins,
              'presets': [], 'ready_for_print': False,
              'limitations': ['Manual approximate bounds; no anatomical anchor registration',
                              'No intersection, clearance or physical thickness certification']}
    report.update(case=args.case, surface_policy='radial_shell_transport_v2' if args.surface_fit else None,
                  scalp_center_m=list(scalp_center), upper_head_width_m=width,
                  formula='p=c+d*(r_scalp(d)+gap+relief*(r_seed-r_ellipsoid(d))); s=width/0.18',
                  render_head='unlit texture; exported head retains original materials')
    if args.surface_fit:
        report['status'] = 'EXPERIMENTAL_RADIAL_SHELL_FIT_NOT_CALIBRATED'
    if args.cage_fit:
        report.update(status='EXPERIMENTAL_SURFACE_DEFORM_CAGE', surface_policy='proxy_surface_deform_v1',
                      formula='C(d)=ray_hit_head(d)+h(d)*normal; bind hair before deforming proxy',
                      caveat='Proxy uses geometric center, not calibrated anatomical landmarks')
    if args.refine_asset:
        report['asset_refinement'] = dict(policy='rear_polar_extension_smooth_vs_voxel_v1',
            rear_extension_radians=.30, close_smooth_iterations=8,
            lift_voxel_size_m_unscaled=.0012, print_certified=False)
    for name, extent, target in presets:
        folder = out / name
        folder.mkdir()
        hair.hide_render = extent is None
        for obj in heads:
            obj.data.materials.clear()
            for mat in original_materials[obj.name]:
                obj.data.materials.append(mat)
        if extent:
            misses = 0
            scale = width / .18
            for v, initial in zip(hair.data.vertices, source):
                v.co = Vector([(initial[k] - center[k]) / size[k] * extent[k] + target[k] for k in range(3)])
                if args.surface_fit:
                    seed_center = Vector((0, .035, .05))
                    delta = (v.co - seed_center) * scale
                    direction = delta.normalized()
                    hit, _, _, distance = scalp_bvh.ray_cast(scalp_center, direction, width * 2)
                    if hit is None:
                        misses += 1
                        v.co = scalp_center + delta
                    else:
                        gap = .004 * scale
                        relief = 1.0 if name == 'surface-volume' else .75
                        radii = (.080 * scale, .090 * scale, .080 * scale)
                        reference = 1 / sum((direction[k] / radii[k]) ** 2 for k in range(3)) ** .5
                        # Signed shell offset preserves inner/outer layers rather than
                        # clamping both layers onto the same scalp surface.
                        radius = distance + gap + relief * (delta.length - reference)
                        v.co = scalp_center + direction * radius
            hair.data.update()
            if args.cage_fit:
                seed_center = Vector((0, .035, .05))
                for v in hair.data.vertices:
                    delta = (v.co - seed_center) * scale
                    if args.refine_asset:
                        radius = delta.length
                        horizontal = math.hypot(delta.x, delta.y)
                        if horizontal > 1e-8 and radius > 1e-8:
                            rear = max(0, delta.y / horizontal)
                            angle = math.acos(max(-1, min(1, delta.z / radius)))
                            angle += .30 * rear * (1 - .5 * max(0, delta.z / radius))
                            delta = Vector((delta.x / horizontal * radius * math.sin(angle),
                                            delta.y / horizontal * radius * math.sin(angle),
                                            radius * math.cos(angle)))
                    v.co = scalp_center + delta
                bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=20)
                cage = bpy.context.object
                cage.name = 'ScalpFitProxy_' + name
                directions = [v.co.normalized() for v in cage.data.vertices]
                for v, d in zip(cage.data.vertices, directions):
                    v.co = scalp_center + Vector((d.x * .080, d.y * .090, d.z * .080)) * scale
                cage.hide_render = True
                cage.display_type = 'WIRE'
                bpy.context.view_layer.objects.active = hair
                modifier = hair.modifiers.new('Reusable scalp proxy', 'SURFACE_DEFORM')
                modifier.target = cage
                bpy.context.view_layer.update()
                bpy.ops.object.surfacedeform_bind(modifier=modifier.name)
                if not modifier.is_bound:
                    raise RuntimeError('Surface deform binding failed')
                misses = 0
                for v, d in zip(cage.data.vertices, directions):
                    hit, normal, _, distance = scalp_bvh.ray_cast(scalp_center, d, width * 2)
                    if hit is None:
                        misses += 1
                        continue
                    if normal.dot(d) < 0:
                        normal = -normal
                    crown = max(0, d.z) ** 2
                    fringe = max(0, -d.y) ** 2 * max(0, d.z)
                    height = (.0015 + .002 * crown) if name == 'cage-close' else (.002 + .009 * crown + .004 * fringe)
                    v.co = hit + normal * height * scale
                cage.data.update()
                bpy.context.view_layer.update()
                # Keep the live binding in a separate working file; evaluate a copy
                # for export so the original hair topology remains available.
                bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'live-cage.blend'))
                evaluated = hair.evaluated_get(bpy.context.evaluated_depsgraph_get())
                deformed = bpy.data.meshes.new_from_object(evaluated)
                hair.modifiers.remove(modifier)
                original_mesh = hair.data
                hair.data = deformed
                report.setdefault('cage', {})[name] = dict(bound=True, proxy_vertices=len(directions),
                    ray_misses=misses, scale=scale, clearance_policy='normal offset, not certified clearance')
                if args.refine_asset:
                    bpy.ops.object.select_all(action='DESELECT')
                    hair.select_set(True)
                    bpy.context.view_layer.objects.active = hair
                    if name == 'cage-lift':
                        remesh = hair.modifiers.new('Hair-only voxel consolidation', 'REMESH')
                        remesh.mode = 'VOXEL'
                        remesh.voxel_size = .0012 * scale
                        remesh.use_smooth_shade = True
                        remesh.use_remove_disconnected = False
                        bpy.ops.object.modifier_apply(modifier=remesh.name)
                    smooth = hair.modifiers.new('Round hair tips', 'SMOOTH')
                    smooth.factor = .5
                    smooth.iterations = 8 if name == 'cage-close' else 4
                    bpy.ops.object.modifier_apply(modifier=smooth.name)
                    for face in hair.data.polygons:
                        face.use_smooth = True
                    # Applying modifiers can replace the mesh datablock.
                    deformed = hair.data
            bpy.ops.object.select_all(action='DESELECT')
            for obj in heads + [hair]:
                obj.select_set(True)
            bpy.ops.export_scene.gltf(filepath=str(folder / 'head-hair.glb'), export_format='GLB',
                                      use_selection=True, export_animations=False)
            bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'editable.blend'))
            if args.surface_fit:
                report.setdefault('ray_misses', {})[name] = misses
        for obj in heads:
            obj.data.materials.clear()
            for mat in review_materials[obj.name]:
                obj.data.materials.append(mat)
        for view, direction in [('front', (0, -1, .03)), ('side', (1, 0, .03)),
                                ('quarter', (.7, -1, .15)), ('back', (0, 1, .03))]:
            camera.location = look + Vector(direction)
            camera.rotation_euler = (look - camera.location).to_track_quat('-Z', 'Y').to_euler()
            scene.render.filepath = str(folder / (view + '.png'))
            bpy.ops.render.render(write_still=True)
        if extent and args.cage_fit:
            hair.data = original_mesh
            bpy.data.meshes.remove(deformed)
            bpy.data.objects.remove(cage, do_unlink=True)
        report['presets'].append({'name': name, 'extent_m': extent, 'center_m': target})
    assert signature(heads) == original
    assert all(digest(root / path) == pin for path, pin in pins.items())
    report['head_geometry_uv_unchanged_in_scene'] = True
    report['source_files_unchanged'] = True
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = []
    for name, _, _ in presets:
        pictures = ''.join(f'<img src="{name}/{view}.png" alt="{view}">' for view in ('front', 'side', 'quarter', 'back'))
        rows.append(f'<h2>{name}</h2><div>{pictures}</div>')
    (out / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Male hair fit trial</title>'
        '<style>body{background:#17202c;color:#eee;font:16px system-ui;margin:24px}div{display:flex}img{width:25%;object-fit:contain}</style>'
        f'<h1>Thử fit tóc nam — {args.case}</h1><p>Base: đầu gốc. So sánh hai mức độ phồng. '
        'Căn theo kích thước thủ công, chưa hiệu chuẩn mốc giải phẫu, chưa kiểm chứng in.</p>' + ''.join(rows), encoding='utf-8')


if __name__ == '__main__':
    main()
