"""Blender subprocess: simplify only the body, retain modular head/hair/hand/cap."""
import argparse
import json
from pathlib import Path
import sys


def main():
    import bpy
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--body-ratio', type=float, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists() or not .1 <= args.body_ratio <= 1:
        raise ValueError('Invalid output or simplification ratio')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source))
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    body = bpy.data.objects['Body_Cuff_Prepared']
    head = bpy.data.objects['Personalized_Complete_Head']
    # Imported flat polygons otherwise duplicate almost every triangle corner
    # during glTF export, making even a decimated body exceed 50 MiB.
    protected_hair = bpy.data.objects.get('Hair_Library_Preset')
    for obj in [body] + ([protected_hair] if protected_hair is not None else []):
        obj.data.polygons.foreach_set('use_smooth', [True]*len(obj.data.polygons))
    bpy.context.view_layer.objects.active = body
    modifier = body.modifiers.new('Web body simplification', 'DECIMATE')
    modifier.ratio = args.body_ratio
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    head_images = set()
    for material in head.data.materials:
        if material and material.use_nodes:
            head_images.update(n.image for n in material.node_tree.nodes
                               if n.type == 'TEX_IMAGE' and n.image)
    resized = []
    for image in bpy.data.images:
        if image in head_images or image.source not in ('FILE', 'GENERATED'):
            continue
        width, height = image.size
        if max(width, height) > 1024:
            # glTF images are lazy-loaded; has_data=False is not an absent image.
            if not image.has_data:
                _ = image.pixels[0]
            factor = 1024 / max(width, height)
            image.scale(max(1, round(width*factor)), max(1, round(height*factor)))
            image.pack()
            resized.append(image.name)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(args.output), export_format='GLB',
        use_selection=True, export_yup=True, export_cameras=False, export_lights=False,
        export_extras=True)
    print(json.dumps({'status': 'ASSEMBLY_WEB_EXPORTED_PENDING_AUDIT',
                      'body_ratio': args.body_ratio, 'non_head_textures_resized': resized}))


if __name__ == '__main__':
    main()
