"""Run-scoped loopback delivery of explicitly selected episode archives."""

import secrets
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def environment_url(environment, episode):
    parts = urlsplit(environment)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "episode"]
    return urlunsplit(parts._replace(query=urlencode([*query, ("episode", episode)])))


class EpisodeServer:
    """One service per run; no directory listing, uploads, or arbitrary path access."""

    def __init__(self, environment):
        parts = urlsplit(environment)
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.paths = {}
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def end_headers(self):
                if self.headers.get("Origin") in (None, owner.origin):
                    self.send_header("Access-Control-Allow-Origin", owner.origin)
                    self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
                    self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Vary", "Origin")
                self.send_header("Cache-Control", "no-store")
                super().end_headers()

            def serve(self, *, body=False, preflight=False):
                origin = self.headers.get("Origin")
                if origin and origin != owner.origin:
                    self.send_error(403)
                    return
                with owner.lock:
                    path = owner.paths.get(self.path)
                if path is None:
                    self.send_error(404)
                    return
                try:
                    stream = path.open("rb")
                except OSError:
                    self.send_error(404)
                    return
                with stream:
                    self.send_response(204 if preflight else 200)
                    self.send_header("Content-Type", "application/zip")
                    if not preflight:
                        self.send_header("Content-Length", str(path.stat().st_size))
                    self.end_headers()
                    if body:
                        try:
                            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
                        except (BrokenPipeError, ConnectionResetError):
                            pass

            def do_GET(self):
                self.serve(body=True)

            def do_HEAD(self):
                self.serve()

            def do_OPTIONS(self):
                self.serve(preflight=True)

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def add(self, path):
        route = f"/{secrets.token_urlsafe(24)}/episode.zip"
        with self.lock:
            self.paths[route] = Path(path).resolve()
        return f"http://127.0.0.1:{self.http.server_port}{route}"

    def __exit__(self, *args):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
