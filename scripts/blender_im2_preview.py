"""Blender > Scripting > Open > Run Script; choose hair.strands.npz in the dialog."""
import bpy
from bpy.props import StringProperty, FloatProperty, IntProperty
from bpy_extras.io_utils import ImportHelper
import numpy as np
from pathlib import Path


class MEMENTO_OT_import_strands(bpy.types.Operator, ImportHelper):
    bl_idname = 'memento.import_strands'
    bl_label = 'Import Im2Haircut strands'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.npz'
    filter_glob: StringProperty(default='*.npz', options={'HIDDEN'})
    radius_ratio: FloatProperty(name='Radius / hair height', default=0.0004,
                               min=0.00001, max=0.01)
    maximum_strands: IntProperty(name='Maximum preview strands (0 = all)', default=0, min=0)
    bevel_resolution: IntProperty(name='Bevel resolution', default=1, min=0, max=3)

    def execute(self, context):
        try:
            with np.load(self.filepath, allow_pickle=False) as data:
                strands = data['strands']
            if (strands.ndim != 3 or strands.shape[2] != 3 or strands.shape[1] < 2
                    or not len(strands) or not np.isfinite(strands).all()):
                raise ValueError('Invalid strands array')
            full_count = len(strands)
            if self.maximum_strands and full_count > self.maximum_strands:
                indices = np.linspace(0, full_count - 1, self.maximum_strands, dtype=int)
                strands = strands[indices]
            # Y is up in the source reference system; do not rotate or rescale silently.
            extent = np.ptp(strands.reshape(-1, 3), axis=0)
            height = float(extent[1])
            if height <= 0:
                raise ValueError('Zero source Y extent')
            curve = bpy.data.curves.new('Im2HaircutStrands', 'CURVE')
            curve.dimensions = '3D'
            curve.resolution_u = 1
            curve.extrude = 0
            curve.bevel_depth = height * self.radius_ratio
            curve.bevel_resolution = self.bevel_resolution
            curve.use_fill_caps = True
            for strand in strands:
                spline = curve.splines.new('POLY')
                spline.points.add(len(strand) - 1)
                coords = np.column_stack((strand, np.ones(len(strand))))
                spline.points.foreach_set('co', coords.ravel())
            material = bpy.data.materials.new('HairPreview_Default_NotRecovered')
            material.diffuse_color = (0.035, 0.02, 0.012, 1)
            material.use_nodes = True
            shader = material.node_tree.nodes.get('Principled BSDF')
            if shader:
                shader.inputs['Base Color'].default_value = material.diffuse_color
                shader.inputs['Roughness'].default_value = 0.5
            curve.materials.append(material)
            obj = bpy.data.objects.new('Im2Haircut_' + Path(self.filepath).stem, curve)
            context.collection.objects.link(obj)
            for selected in context.selected_objects:
                selected.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            obj['source_file'] = self.filepath
            obj['source_strand_count'] = full_count
            obj['preview_strand_count'] = len(strands)
            obj['material_is_recovered'] = False
            obj['units_calibrated'] = False
            self.report({'INFO'}, f'{len(strands)} strands; bevel radius {curve.bevel_depth:.6g} source units')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


previous = getattr(bpy.types, 'MEMENTO_OT_import_strands', None)
if previous is not None:
    bpy.utils.unregister_class(previous)
bpy.utils.register_class(MEMENTO_OT_import_strands)
bpy.ops.memento.import_strands('INVOKE_DEFAULT')
