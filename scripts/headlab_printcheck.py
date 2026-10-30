"""Conservative geometry screening and optional external manufacturing driver."""
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import trimesh
from headlab_algorithms import normal_ray_screen


def orientation_screen(mesh):
    # A fixed search policy shared by every candidate; score is a heuristic only.
    rows=[];areas=mesh.area_faces;normals=mesh.face_normals
    for tilt in [20,35,50]:
        for yaw in [0,180]:
            transform=trimesh.transformations.euler_matrix(np.deg2rad(tilt),0,np.deg2rad(yaw),'sxyz')
            n=normals@transform[:3,:3].T
            down=np.maximum(0,-n[:,2]-.35)
            # Original source is STL +Z up, face points -Y. Protect its front zone.
            front=np.clip(-normals[:,1],0,1)
            cost=float(np.sum(areas*down*(1+2*front)))
            rows.append({'tilt_x_deg':tilt,'yaw_z_deg':yaw,'weighted_downward_area_proxy_mm2':cost})
    best=min(rows,key=lambda x:x['weighted_downward_area_proxy_mm2'])
    return {'status':'HEURISTIC_ONLY','best':best,'candidates':rows,'supports_generated':False,'orientation_verified':False,'scope':'area/normal proxy, not support strength, peel-force or sliced-layer validation'}


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def external_driver(stl,output,profile,driver):
    """Driver owns support/slicing/analysis. No shell, no fake default pass.

    args placeholders: {input}, {output}, {profile}. Outputs must be fresh,
    report.json must bind input/profile hashes, machine artifact and checks.
    Physical print success is never inferred from a driver report.
    """
    if not driver or not driver.get('command'):
        return {'status':'NOT_CONFIGURED','ready_for_machine':False,'reason':'No support/slicing driver, resin calibration or machine artifact verified'}
    output.mkdir()
    command=driver['command']
    if not isinstance(command,list) or not command or not all(isinstance(x,str) for x in command):raise ValueError('Driver command must be a nonempty argv array')
    mapping={'input':str(stl.resolve()),'output':str(output.resolve()),'profile':str(profile.resolve())}
    argv=[]
    for argument in command:
        for key,value in mapping.items():argument=argument.replace('{'+key+'}',value)
        argv.append(argument)
    run=subprocess.run(argv,shell=False,capture_output=True,text=True,timeout=driver.get('timeout_seconds',180))
    (output/'stdout.log').write_text(run.stdout,encoding='utf-8');(output/'stderr.log').write_text(run.stderr,encoding='utf-8')
    if run.returncode:raise ValueError('Manufacturing driver failed: '+str(run.returncode))
    report=json.loads((output/'report.json').read_text(encoding='utf-8-sig'))
    if report.get('input_sha256')!=file_hash(stl) or report.get('profile_sha256')!=file_hash(profile):raise ValueError('Driver provenance mismatch')
    required=['supports_checked','sliced_layers_checked','machine_format_checked']
    if not all(report.get(x) is True for x in required):raise ValueError('Driver did not verify all manufacturing checks')
    artifact=(output/report['machine_artifact']).resolve()
    if not artifact.is_relative_to(output.resolve()) or not artifact.is_file() or artifact.stat().st_size==0:raise ValueError('Missing or unsafe driver artifact')
    if report.get('machine_artifact_sha256')!=file_hash(artifact):raise ValueError('Machine artifact checksum mismatch')
    calibration=json.loads(profile.read_text(encoding='utf-8-sig'))
    valid=calibration.get('calibration_status')=='PHYSICAL_PROFILE_VALIDATED'
    return {'status':'DRIVER_VERIFIED' if valid else 'DRIVER_CHECKED_UNCALIBRATED_PROFILE','ready_for_machine':valid,'driver_report':report,'physical_print_verified_for_this_model':False}


def screen(stl,profile):
    mesh=trimesh.load(stl,force='mesh');started=time.perf_counter()
    finite=bool(np.isfinite(mesh.vertices).all());components=len(mesh.split(only_watertight=False))
    return {'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'finite':finite,
        'connected_components':components,'triangles':len(mesh.faces),'volume_mm3':float(mesh.volume),'extents_mm':mesh.extents.tolist(),
        'topology_pass':bool(finite and mesh.is_watertight and mesh.is_winding_consistent and components==1 and mesh.volume>0),
        'normal_ray_screen':normal_ray_screen(mesh,profile['ray_samples'],profile['trial_feature_mm']),
        'orientation':orientation_screen(mesh),'self_intersections_verified':False,'minimum_wall_thickness_verified':False,
        'slicer_verified':False,'physical_print_verified':False,'screen_seconds':time.perf_counter()-started,'ready_for_machine':False}
