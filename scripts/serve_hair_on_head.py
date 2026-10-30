"""Project-root local viewer shim."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from memento3d_hair_on_head.entrypoint import serve
if __name__=='__main__':serve()
