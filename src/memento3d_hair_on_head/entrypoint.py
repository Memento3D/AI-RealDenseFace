"""Locate the project by package location, independent of the caller's cwd."""
from pathlib import Path
import sys,runpy

def project_root():return Path(__file__).resolve().parents[2]

def activate_runtime():
    package=Path(__file__).resolve().parent
    for folder in [package/'vendor',package/'runtime']:
        value=str(folder)
        if value not in sys.path:sys.path.insert(0,value)

def cli():
    activate_runtime()
    from .runtime.pipeline import main
    return main()

def serve():
    runpy.run_path(str(Path(__file__).resolve().parent/'runtime/serve_review.py'),run_name='__main__')
