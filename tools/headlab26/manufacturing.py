"""Configured slicer execution plus checks on actual exported raster layers."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from .io import write,sha
from .process import execute

def check_layers(folder,max_pixels=40_000_000):
 files=sorted(Path(folder).glob('*.png'))
 if not files:raise ValueError('No actual sliced PNG layers found')
 previous=None;records=[];sizes=None;empty=0
 for layer,path in enumerate(files):
  with Image.open(path) as image:
   if image.width*image.height>max_pixels:raise ValueError('Layer pixel budget exceeded')
   current=np.asarray(image.convert('L'))>127
  if sizes is not None and current.shape!=sizes:raise ValueError('Slice dimensions change across layers')
  sizes=current.shape;labels,count=ndi.label(current);unattached=0
  for i in range(1,count+1):
   region=labels==i
   if previous is not None and not (region&previous).any():unattached+=1
  if unattached:records.append({'layer':layer,'new_components_without_previous_pixel_overlap':unattached})
  if not current.any():empty+=1
  previous=current
 return {'status':'ACTUAL_RASTER_LAYERS_ANALYZED','layer_count':len(files),'dimensions':list(sizes),'empty_layers':empty,'island_layer_flags':records,'scope':'thresholded pixel-overlap connectivity, not support strength or resin-trap proof','physical_print_verified':False}

def run(stl,out,settings,project):
 out=Path(out);out.mkdir(exist_ok=False,parents=True);values={'input_stl':str(Path(stl).resolve()),'output_dir':str(out.resolve()),'layers_dir':str((out/'layers').resolve()),'machine_file':str((out/'machine_output').resolve())};report={'status':'SLICER_NOT_CONFIGURED','ready_for_machine':False,'physical_print_verified':False,'stages':[]}
 if not settings.get('slicer_argv'):write(out/'report.json',report);return report
 try:
  (out/'layers').mkdir()
  for label,field in [('slice','slicer_argv'),('machine_export','machine_export_argv'),('uvtools','uvtools_argv')]:
   argv=settings.get(field,[])
   if not argv:continue
   if not isinstance(argv,list) or not all(isinstance(x,str) for x in argv):raise ValueError('Commands must be argv arrays, never shell text')
   command=[arg.format_map(values) for arg in argv];row=execute(label,command,project,out,1800);report['stages'].append(row)
   if row['exit_code']!=0:raise ValueError(label+' process failed')
  report['layers']=check_layers(out/'layers');machine=Path(values['machine_file']);report['machine_file_exists']=machine.is_file()
  if machine.is_file():report['machine_file_sha256']=sha(machine)
  report['status']='SLICED_ARTIFACTS_EXPORTED_UNVERIFIED'
  report['readiness_blockers']=['physical process and supports not independently verified']
  if report['layers']['island_layer_flags']:report['readiness_blockers'].append('actual layer connectivity flags')
  if not settings.get('calibrated_profile'):report['readiness_blockers'].append('machine/resin profile uncalibrated')
  if not settings.get('machine_format_verified'):report['readiness_blockers'].append('exact machine file format unverified')
 except Exception as exc:report['status']='MANUFACTURING_STAGE_FAILED';report['error']=str(exc)
 write(out/'report.json',report);return report
