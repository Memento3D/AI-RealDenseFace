"""Blender adapter: experimental root shell and hair-only consolidation.

Consumes a precomputed scalp patch; never edits the source head or drops islands.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import bmesh
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-blend', required=True)
    parser.add_argument('--patch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    source, patch = Path(args.source_blend).resolve(), Path(args.patch).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    pins = {str(p): sha(p) for p in (source, patch)}
    patch_data = np.load(patch, allow_pickle=False)
    report = {'status': 'ROOT_SHELL_CONSOLIDATION_EXPERIMENT', 'inputs': pins,
              'ready_for_print': False, 'variants': []}
    for name, thickness, voxel in [('thin-root', .003, .0012), ('strong-root', .005, .0015)]:
        bpy.ops.wm.open_mainfile(filepath=str(source))
        scene = bpy.context.scene
        heads = [o for o in scene.objects if o.type == 'MESH' and len(o.data.polygons) == 9976]
        if len(heads) != 1:
            raise RuntimeError('Expected exactly one frozen head mesh')
        head = heads[0]
        hairs = [o for o in scene.objects if o.type == 'MESH' and o != head and not o.hide_render]
        if len(hairs) != 1:
            raise RuntimeError('Ambiguous source hair mesh')
        hair = hairs[0]
        before = [(tuple(v.co)) for v in head.data.vertices]
        folder = out / name
        folder.mkdir()
        mesh = bpy.data.meshes.new('Scalp patch from frozen buzz mask')
        mesh.from_pydata(patch_data['vertices'].tolist(), [], patch_data['faces'].tolist())
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        cap = bpy.data.objects.new('HairRootShell', mesh)
        scene.collection.objects.link(cap)
        bpy.context.view_layer.objects.active = cap
        bpy.ops.object.select_all(action='DESELECT')
        cap.select_set(True)
        solid = cap.modifiers.new('Root shell thickness', 'SOLIDIFY')
        solid.thickness = thickness
        solid.offset = 0
        solid.use_even_offset = True
        bpy.ops.object.modifier_apply(modifier=solid.name)
        cap.data.materials.append(hair.data.materials[0])
        hair.select_set(True)
        bpy.context.view_layer.objects.active = hair
        bpy.ops.object.join()
        remesh = hair.modifiers.new('Union hair and root via voxel', 'REMESH')
        remesh.mode = 'VOXEL'
        remesh.voxel_size = voxel
        remesh.use_remove_disconnected = False
        remesh.use_smooth_shade = True
        bpy.ops.object.modifier_apply(modifier=remesh.name)
        smooth = hair.modifiers.new('Soften voxel surface', 'SMOOTH')
        smooth.factor, smooth.iterations = .35, 3
        bpy.ops.object.modifier_apply(modifier=smooth.name)
        for face in hair.data.polygons:
            face.use_smooth = True
        bpy.ops.object.select_all(action='DESELECT')
        head.select_set(True)
        hair.select_set(True)
        bpy.ops.export_scene.gltf(filepath=str(folder / 'head-hair.glb'), export_format='GLB',
                                  use_selection=True, export_animations=False)
        bpy.ops.object.select_all(action='DESELECT')
        hair.select_set(True)
        bpy.ops.export_scene.gltf(filepath=str(folder / 'hair-only.glb'), export_format='GLB',
                                  use_selection=True, export_animations=False)
        bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'editable.blend'))
        assert before == [tuple(v.co) for v in head.data.vertices]
        # Review-only emission material, after exporting original head materials.
        for slot in head.material_slots:
            mat = slot.material.copy()
            slot.material = mat
            tree = mat.node_tree
            tex = next(n for n in tree.nodes if n.type == 'TEX_IMAGE')
            output = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
            emission = tree.nodes.new('ShaderNodeEmission')
            tree.links.new(tex.outputs['Color'], emission.inputs['Color'])
            tree.links.new(emission.outputs[0], output.inputs['Surface'])
        from mathutils import Vector
        look = Vector((0, .035, .035))
        for view, direction in [('front', (0, -1, .03)), ('side', (1, 0, .03)),
                                ('quarter', (.7, -1, .15)), ('back', (0, 1, .03))]:
            camera = scene.camera
            camera.location = look + Vector(direction)
            camera.rotation_euler = (look-camera.location).to_track_quat('-Z', 'Y').to_euler()
            scene.render.filepath = str(folder / (view+'.png'))
            bpy.ops.render.render(write_still=True)
        report['variants'].append(dict(name=name, root_thickness_model_units=thickness,
            voxel_model_units=voxel, dropped_components=False))
    assert all(sha(Path(p)) == value for p, value in pins.items())
    report['source_files_unchanged'] = True
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
