"""Vercel function: serves the visualizer API at /api/run.

Vercel maps every .py file in /api to a route (api/run.py -> /api/run) and
looks for a class named `handler` that inherits from BaseHTTPRequestHandler.

The whole project is bundled with the function, so transformer.py sits right
next to it and imports cleanly.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_api import handle_params  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") not in ("/api/run", "/run"):
            self.send_error(404, "Not found")
            return

        query = {
            key: values[0]
            for key, values in parse_qs(
                parsed.query, keep_blank_values=True
            ).items()
        }
        status, body = handle_params(query)
        raw = json.dumps(body).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))
