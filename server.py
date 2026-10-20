"""Lightweight visualization server for transformer.py.

Standard library only - no pip installs required.

    python server.py            # http://127.0.0.1:8000
    python server.py --port 9000
"""

import argparse
import json
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

sys.path.insert(0, BASE_DIR)

from transformer_api import handle_params  # noqa: E402


class VisualizerHandler(SimpleHTTPRequestHandler):
    server_version = "TransformerVisualizer/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            self.handle_api_run(parsed)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        # never let the browser hold on to a stale app.js / styles.css
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def handle_api_run(self, parsed):
        query = {
            key: values[0]
            for key, values in parse_qs(
                parsed.query, keep_blank_values=True
            ).items()
        }
        status, body = handle_params(query)
        self.send_json(body, status)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        path = getattr(self, "path", "") or ""
        if path.startswith("/api/"):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(
        description="Serve the transformer visualizer UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not os.path.isdir(STATIC_DIR):
        raise SystemExit(f"Missing static directory: {STATIC_DIR}")

    handler = partial(VisualizerHandler, directory=STATIC_DIR)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Transformer visualizer running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
