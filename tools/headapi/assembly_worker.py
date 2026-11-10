"""Per-job head + pinned modular body/hand/cap + library hair preview.

No experimental person's head is published as the new person's reconstruction.
The reference bundle supplies only size/eye anchors for a fixed design preset.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
from .demo_scene_worker import inspect_glb
from .service import load_config, sha, write

POLICY = 'graduation_modular_preview_v1'


def asset_paths(config):
    root = Path(config['root'])
    return {name: (root / entry['path']).resolve()
            for name, entry in config['assembly']['assets'].items()}


def check_assets(config):
    """No SDK imports are needed by the API preflight; see assembly_config.py."""
    from .assembly_config import validate_assets
    return validate_assets(config)


def world_mesh(scene, node):
    matrix, geometry = scene.graph[node]
    mesh = scene.geometry[geometry].copy()
    mesh.apply_transform(matrix)
    return mesh


def compose(job, output, config=None):
    import numpy as np
    import trimesh
    from .assembly_math import AXES, eye_alignment, fit_library_hair
    from .assembly_cohesion_trial import boundary_loops, bridge_rings, smooth_seam_normals
    from .head_assembly_trial import condition_neck
    job, output = Path(job).resolve(), Path(output).resolve()
    config = config or load_config(job / 'config.snapshot.json')
    check_assets(config)
    paths = asset_paths(config)
    output.mkdir(parents=True, exist_ok=False)
    uv = json.loads((job / 'rdf_uv/report.json').read_text(encoding='utf-8'))
    head_path = job / 'rdf_uv/final/face_textured.glb'
    bundle = job / 'rdf/projection_bundle.npz'
    if uv.get('status') != 'UV_FACE_EXPORTED' or uv.get('head_sha256') != sha(head_path):
        raise ValueError('Per-job UV head is not verified')
    if uv.get('input_sha256', {}).get(str(bundle)) != sha(bundle):
        raise ValueError('UV head does not belong to this job bundle')
    # Use the existing generic appearance pass, not a cached research atlas.
    from .assembly_appearance import run as appearance
    appearance(job, output / 'appearance', paths)
    head_path = output / 'appearance/head.glb'
    with np.load(bundle, allow_pickle=False) as data:
        vertices, faces = data['vertices'], data['faces']
    with np.load(paths['reference_bundle'], allow_pickle=False) as data:
        reference = data['vertices']
        if not np.array_equal(faces, data['faces']):
            raise ValueError('Preset reference topology differs from current RDF')
    with np.load(paths['embedding'], allow_pickle=False) as data:
        embedding = {name: data[name] for name in ('full_faces', 'full_weights')}
    head = trimesh.load(head_path, force='mesh', process=False)
    if not np.array_equal(head.vertices[head.faces].astype(np.float32), vertices[faces].astype(np.float32)):
        raise ValueError('Personalized head does not preserve the current job geometry')
    pose = json.loads(paths['reference_pose'].read_text(encoding='utf-8'))
    matrix = eye_alignment(vertices, faces, embedding, reference,
                           pose['cases']['balanced']['matrix_working'])
    # Work in the same Z-up frame as the neck/cap experiments.
    head.apply_transform(matrix)
    with paths['semantic_masks'].open('rb') as stream:
        neck_ids = pickle.load(stream, encoding='latin1')['neck']
    # Bind semantic indices through triangle corners, not GLB vertex indices.
    corner_neck = np.isin(faces.ravel(), neck_ids)
    mask = np.zeros(len(head.vertices), dtype=bool)
    np.logical_or.at(mask, head.faces.ravel(), corner_neck)
    before = head.vertices.copy()
    head.vertices = condition_neck(head.vertices, mask)
    if not np.array_equal(before[~mask], head.vertices[~mask]):
        raise ValueError('Personalized non-neck geometry changed')
    cut_height = config['assembly']['neck_cut_z']
    hv, hf, hu = trimesh.intersections.slice_faces_plane(
        head.vertices, head.faces, [0, 0, 1], [0, 0, cut_height], uv=head.visual.uv)
    head = trimesh.Trimesh(hv, hf, process=False,
        visual=trimesh.visual.TextureVisuals(uv=hu, material=head.visual.material))
    head.vertex_normals = smooth_seam_normals(head.vertices, head.faces)
    template = trimesh.load(paths['template'], force='scene', process=False)
    result = trimesh.Scene()
    retained = []
    for node in config['assembly']['retained_nodes']:
        mesh = world_mesh(template, node)
        mesh.apply_transform(AXES)
        result.add_geometry(mesh, node_name=node, geom_name=node)
        retained.append(node)
    # Rebuild the neck strip for this head; the old head-specific strip is excluded.
    old_neck = world_mesh(template, 'Neck_Transition')
    old_neck.apply_transform(AXES)
    body_loops = boundary_loops(old_neck.vertices, old_neck.faces)
    if len(body_loops) != 2:
        raise ValueError('Expected two neck transition boundary rings')
    body_ring = min(body_loops, key=lambda ring: ring[:, 2].mean())
    head_loops = boundary_loops(head.vertices, head.faces)
    neck_loops = [ring for ring in head_loops if np.max(np.abs(ring[:, 2] - cut_height)) < 1e-5]
    if len(neck_loops) != 1:
        raise ValueError('Expected one clipped neck ring')
    nv, nf = bridge_rings(body_ring, neck_loops[0])
    neck = trimesh.Trimesh(nv, nf, process=False)
    # Color is a display estimate sampled from this head's lower-neck texture.
    image = np.asarray(head.visual.material.baseColorTexture.convert('RGB'))
    sample_uv = head.visual.uv[head.vertices[:, 2] < cut_height + .02]
    if not len(sample_uv):
        raise ValueError('No neck texels available')
    x = np.clip((sample_uv[:, 0] * image.shape[1]).astype(int), 0, image.shape[1]-1)
    y = np.clip(((1-sample_uv[:, 1]) * image.shape[0]).astype(int), 0, image.shape[0]-1)
    color = np.r_[np.median(image[y, x], axis=0), 255].astype(np.uint8)
    neck.visual.vertex_colors = np.tile(color, (len(nv), 1))
    result.add_geometry(neck, node_name='Neck_Transition', geom_name='Neck_Transition')
    result.add_geometry(head, node_name='Personalized_Complete_Head', geom_name='Personalized_Complete_Head')
    from .assembly_config import base_only
    hair_scale = None
    optional_components = []
    if not base_only(config):
        library = trimesh.load(paths['hair_library'], force='scene', process=False)
        hair = world_mesh(library, 'Male_Hair')
        library_head = world_mesh(library, 'geometry_0')
        hair.vertices, hair_scale = fit_library_hair(hair.vertices, library_head.vertices, vertices)
        hair.apply_transform(matrix)
        # Crop hidden hair above the cap board. This is preview-only, not a closed print mesh.
        board = result.geometry['Cap_Board']
        top = float(board.bounds[0, 2] - .001)
        # Low-level uncapped clipping avoids an unnecessary Shapely dependency.
        vh, fh, uh = trimesh.intersections.slice_faces_plane(
            hair.vertices, hair.faces, [0, 0, -1], [0, 0, top],
            uv=getattr(hair.visual, 'uv', None))
        hair = trimesh.Trimesh(vh, fh, process=False,
            visual=trimesh.visual.TextureVisuals(uv=uh, material=hair.visual.material))
        if not len(hair.faces):
            raise ValueError('Hair is entirely outside the cap preset')
        result.add_geometry(hair, node_name='Hair_Library_Preset', geom_name='Hair_Library_Preset')
        optional_components.append('Hair_Library_Preset')
    result.apply_transform(np.linalg.inv(AXES))
    master = output / 'assembly-master.glb'
    result.export(master)
    inspect_glb(master)
    report = {'policy': POLICY, 'status': 'ASSEMBLY_EXPORTED_PENDING_WEB_AUDIT',
        'job_id': json.loads((job/'job.json').read_text(encoding='utf-8'))['job_id'],
        'input_sha256': {str(p): sha(p) for p in [bundle, head_path, *paths.values()]},
        'matrix_native_to_working': matrix.tolist(), 'neck_cut_z': cut_height,
        'components': retained + ['Neck_Transition', 'Personalized_Complete_Head'] + optional_components,
        'output_kind': 'customization_base' if base_only(config) else 'assembled_preview',
        'customization_ready': False,
        'local_neck_lip_baseline_integrated': False,
        'hair_fit_axis_scale': hair_scale.tolist() if hair_scale is not None else None,
        'hair_is_designed_preset': hair_scale is not None,
        'old_personalized_head_included': False, 'non_neck_shape_preserved_up_to_similarity': True,
        'master_sha256': sha(master), 'ready_for_machine': False,
        'limitations': ['Preset hairstyle, not inferred from photos',
            'Neck color estimated; connector is a separate component',
            'Hair hidden above cap is clipped; no mesh union or print certification',
            'Hand/cap contact comes from the fixed preset; full collision audit pending',
            'Base-only mode is a technical preview; accepted local neck/lip fixes are not yet generalized',
            'No published compatible asset catalog or per-head accessory placements are included']}
    write(output/'assembly-report.json', report)
    return report


def verify_export(output, config):
    output = Path(output)
    report = json.loads((output/'assembly-report.json').read_text(encoding='utf-8'))
    if sha(output/'assembly-master.glb') != report['master_sha256']:
        raise ValueError('Assembly master changed after composition')
    scene = output/'scene.glb'
    check = inspect_glb(scene)
    if check['bytes'] >= 50*1024**2 or not check['embedded_base_color_texture']:
        raise ValueError('Assembly exceeds backend GLB contract or lacks texture')
    from .assembly_audit import audit
    report['web_audit'] = audit(output/'assembly-master.glb', scene, report['components'])
    check_assets(config)
    report.update(status='UV_SCENE_EXPORTED', scene_sha256=sha(scene),
                  presentation_policy=config.get('presentation_policy', 'none'))
    write(output/'report.json', report)
    return report


def run(job, output, config_path=None):
    from .runtime_budget import run_guarded
    job, output = Path(job).resolve(), Path(output).resolve()
    config = load_config(config_path or job/'config.snapshot.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Release all NumPy/Trimesh buffers before starting Blender. The API and
    # orchestration process never retain a second full copy of the body mesh.
    compose_command = [sys.executable, '-B', '-X', 'utf8', '-m',
        'tools.headapi.assembly_worker', '--job', str(job), '--output', str(output), '--compose-only']
    if config_path:
        compose_command.extend(['--config', str(config_path)])
    with output.with_suffix('.compose.log').open('x', encoding='utf-8') as log:
        done = run_guarded(compose_command, cwd=config['root'], env=None, stdout=log,
            timeout=config.get('stage_timeout_seconds', 1800), minimum=2*1024**3, abort=768*1024**2)
    if done.returncode:
        raise ValueError('Assembly composition failed; see '+output.with_suffix('.compose.log').name)
    write(output/'assembly-config.snapshot.json', config)
    with (output/'web-export.log').open('x', encoding='utf-8') as log:
        command = [config['paths']['blender'], '--background', '--factory-startup',
            '--disable-autoexec', '--python-exit-code', '1', '-t', '4', '--python',
            str(Path(config['root'])/'tools/headapi/assembly_web_export.py'), '--',
            '--source', str(output/'assembly-master.glb'), '--output', str(output/'scene.glb'),
            '--body-ratio', str(config['assembly']['web_body_ratio'])]
        done = run_guarded(command, cwd=config['root'], env=None, stdout=log,
            timeout=config.get('stage_timeout_seconds', 1800), minimum=2*1024**3, abort=768*1024**2)
    if done.returncode:
        raise ValueError('Assembly web export failed; see web-export.log')
    return verify_export(output, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--presentation-policy')  # shared scene-worker command contract
    parser.add_argument('--config', type=Path, help='Explicit config for offline artifact replay only')
    parser.add_argument('--compose-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        verify_export(args.output, load_config(args.config or args.job/'config.snapshot.json'))
    elif args.compose_only:
        compose(args.job, args.output, load_config(args.config) if args.config else None)
    else:
        run(args.job, args.output, args.config)
