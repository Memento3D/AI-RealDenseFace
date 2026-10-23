"""Run the integrated CPU head workspace without reinstalling dependencies."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from memento3d_head.bootstrap import main
if __name__=='__main__':raise SystemExit(main(['optimize-head',*sys.argv[1:]]))
