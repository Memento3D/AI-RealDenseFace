"""Standard-library admission and pins for the opt-in assembly pipeline."""
from pathlib import Path
from .service import sha

POLICY = 'graduation_modular_preview_v1'


def base_only(config):
    return config.get('assembly', {}).get('output_mode', 'assembled_preview') == 'customization_base'


def validate_assets(config):
    assembly = config.get('assembly', {})
    if assembly.get('policy') != POLICY or config.get('face_render_policy') != 'strict_uv_v1':
        raise ValueError('Assembly requires a supported policy and strict UV head')
    if config.get('presentation_policy') != 'harmonized_head_v1':
        raise ValueError('Assembly requires the per-job appearance policy')
    mode = assembly.get('output_mode', 'assembled_preview')
    if mode not in ('assembled_preview', 'customization_base'):
        raise ValueError('Unsupported assembly output mode')
    required = {'template', 'hair_library', 'reference_bundle', 'reference_pose',
                'embedding', 'semantic_masks', 'uv_assets'}
    assets = assembly.get('assets', {})
    if set(assets) != required:
        raise ValueError('Assembly assets are incomplete')
    for name, entry in assets.items():
        path = Path(config['root']) / entry['path']
        if not path.is_file() or sha(path) != entry['sha256']:
            raise ValueError('Assembly asset pin changed or missing: ' + name)
    blender = Path(config['paths'].get('blender', ''))
    if not blender.is_file():
        raise ValueError('Assembly requires Blender')
    if not .1 <= assembly.get('web_body_ratio', 0) <= 1:
        raise ValueError('Invalid assembly web body ratio')
    if not .75 <= assembly.get('neck_cut_z', 0) <= .80:
        raise ValueError('Invalid assembly neck cut')
    expected = ['Body_Cuff_Prepared', 'Wrist_Lining_Join', 'Complete_Hand_Posed_Static',
                'Cap_Cup', 'Cap_Board', 'Cap_Tassel_New', 'Cap_Cord_New']
    if base_only(config):
        expected = expected[:3]
    if assembly.get('retained_nodes') != expected:
        raise ValueError('Unsupported modular component selection')
    return True
