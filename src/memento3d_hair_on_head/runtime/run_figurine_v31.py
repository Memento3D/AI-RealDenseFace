"""V3.1 entry point; current project root and existing V3 CPU dependencies."""
import sys
import run_figurine_v3 as runner
import figurine_preview
if __name__=='__main__':
    if '3.1' not in getattr(runner,'SUPPORTED_FIGURINE_DESIGN_VERSIONS',()):
        raise SystemExit('Incomplete V3.1 update: replace scripts/run_figurine_v3.py from the update ZIP.')
    if not getattr(figurine_preview,'TEXTURED_EYE_SUPPORT',False):
        raise SystemExit('Incomplete V3.1 update: replace scripts/figurine_preview.py from the update ZIP.')
    if '--profile' not in sys.argv:sys.argv[1:1]=['--profile',str(runner.ROOT/'configs/figurine04.v31.json')]
    runner.main()
