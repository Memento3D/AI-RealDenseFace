"""Loopback-only generated-result viewer. No model fitting or file writes over HTTP."""
import argparse,json,threading,functools
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();folder=a.directory.resolve()
if not (folder/'head_with_hair.glb').is_file():raise SystemExit('Missing assembled head')
class Handler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(folder)))
url='http://127.0.0.1:'+str(server.server_address[1])+'/viewer/index.html'
(folder/'review_server.json').write_text(json.dumps({'url':url,'loopback_only':True}))
timer=threading.Timer(3600,server.shutdown);timer.daemon=True;timer.start()
try:server.serve_forever()
finally:server.server_close()
