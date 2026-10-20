"""Vercel function: the visualizer API and the static site.

Vercel routes every request to this single function, so it does both jobs:
  * /api/run?text=...  -> the JSON payload with every intermediate tensor
  * everything else     -> the static files from static/

The project is bundled with the function, so transformer.py sits right next to
it and imports cleanly.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT, "static")

sys.path.insert(0, ROOT)

from transformer_api import handle_params  # noqa: E402

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
}

# query keys the front end only ever sends to the API
API_KEYS = ("embed_dim", "hidden_dim", "max_seq_len", "seed")


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = {
            key: values[0]
            for key, values in parse_qs(
                parsed.query, keep_blank_values=True
            ).items()
        }

        if self.is_api_request(parsed.path, query):
            self.serve_api(query)
        else:
            self.serve_static(parsed.path)

    def do_HEAD(self):
        self.do_GET()

    def is_api_request(self, path, query):
        if "api/run" in path or path.rstrip("/") in ("/run", "/api"):
            return True
        # fallback: the front end always sends the tuning parameters
        return path in ("", "/") and any(key in query for key in API_KEYS)

    def serve_api(self, query):
        status, body = handle_params(query)
        raw = json.dumps(body).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def serve_static(self, path):
        relative = path.lstrip("/") or "index.html"
        target = os.path.normpath(os.path.join(STATIC_DIR, relative))

        if not target.startswith(STATIC_DIR) or not os.path.isfile(target):
            self.send_error(404, "Not found")
            return

        with open(target, "rb") as handle:
            data = handle.read()

        extension = os.path.splitext(target)[1].lower()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            CONTENT_TYPES.get(extension, "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header(
            "Cache-Control",
            "no-store" if extension in (".html",) else "public, max-age=3600",
        )
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))
