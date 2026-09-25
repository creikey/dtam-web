# Dev server for web/ with caching disabled: python3 web/learn/tools/serve.py 8767
import http.server, sys, functools, os

class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

root = os.path.join(os.path.dirname(__file__), "..", "..")
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8767
http.server.ThreadingHTTPServer(("", port), functools.partial(NoCache, directory=root)).serve_forever()
