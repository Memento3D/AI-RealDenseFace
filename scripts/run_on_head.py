"""Project-root CLI shim; implementation lives in src/memento3d_hair_on_head."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from memento3d_hair_on_head.entrypoint import cli
if __name__=='__main__':raise SystemExit(cli())

# Compatibility for the feature's earlier unittest/import API.
def run(*args,**kwargs):
    from memento3d_hair_on_head.entrypoint import activate_runtime
    activate_runtime()
    from memento3d_hair_on_head.runtime.pipeline import run as implementation
    return implementation(*args,**kwargs)
