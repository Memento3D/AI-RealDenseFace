"""V3.2: trace detected masks onto the source head; retain V3.1 eye rendering."""
import sys
import run_figurine_v3 as runner
if __name__=='__main__':
    if '3.2' not in getattr(runner,'SUPPORTED_FIGURINE_DESIGN_VERSIONS',()):
        raise SystemExit('Incomplete V3.2 update: replace scripts/run_figurine_v3.py from the update ZIP.')
    if '--profile' not in sys.argv:sys.argv[1:1]=['--profile',str(runner.ROOT/'configs/figurine04.v32.json')]
    runner.main()
