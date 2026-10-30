"""V2.2 experiment: refined texture and separated existing FLAME eyes."""
import sys
from run_v1 import main

if __name__=='__main__':
    if '--refine-texture' not in sys.argv:
        sys.argv.append('--refine-texture')
    if '--separate-eyes' not in sys.argv:
        sys.argv.append('--separate-eyes')
    raise SystemExit(main())
