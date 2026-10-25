"""V2.1 experiment: same reconstruction as V1, refined final texture bake."""
import sys
from run_v1 import main

if __name__=='__main__':
    if '--refine-texture' not in sys.argv:
        sys.argv.append('--refine-texture')
    raise SystemExit(main())
