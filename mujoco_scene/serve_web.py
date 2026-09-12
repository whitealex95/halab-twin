#!/usr/bin/env python3
"""Serve the self-contained HaLab viewer locally, without external services."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import argparse,functools
ROOT=Path(__file__).resolve().parent
class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-cache')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()
    def log_message(self,format,*args):
        if args and str(args[1] if len(args)>1 else '')!='200':super().log_message(format,*args)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    if not (ROOT/'web/data/manifest.json').exists():p.error('Run python mujoco_scene/scripts/export_web.py first.')
    server=ThreadingHTTPServer((a.host,a.port),functools.partial(Handler,directory=str(ROOT/'web')))
    print(f'HaLab viewer: http://{a.host}:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
