"""Stage-specific source bindings: changing hair code never invalidates a face fit."""
from pathlib import Path
from .io import key,sha

STAGE_FILES={
 'prepare':['tools/headlab26/prepare.py','tools/headlab26/io.py','scripts/head24_detection.py','scripts/detect_head24_profile.py','scripts/head24_policy.py','head24-rdf-source.json'],
 'semantic':['tools/headlab26/semantics.py','tools/headlab26/io.py','src/memento3d_head/adapters/automatic_face_masks.py'],
 'fit':['tools/headlab26/rdf_worker.py','tools/headlab26/policy.py','tools/headlab26/io.py','scripts/head24_policy.py','scripts/eye_components.py','head24-rdf-source.json'],
}

def stage_code(root,stage):
 root=Path(root)
 return key({'binding_code':sha(Path(__file__)),'files':{name:sha(root/name) for name in STAGE_FILES[stage]}})
