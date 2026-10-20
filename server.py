"""Lightweight visualization server for transformer.py.

Standard library only - no pip installs required.

    python server.py            # http://127.0.0.1:8000
    python server.py --port 9000
"""

import argparse
import json
import os
import sys
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

sys.path.insert(0, BASE_DIR)

from transformer import TransformerPipeline  # noqa: E402

DEFAULT_TEXT = "akshay is running the dog is barking"

MAX_TEXT_LENGTH = 500


def build_payload(text, embed_dim, hidden_dim, max_seq_len, seed):
    pipeline = TransformerPipeline(
        text=text,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    return pipeline.run()


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
        query = parse_qs(parsed.query)

        if "text" in query:
            text = (query.get("text", [""])[0] or "").strip()
            if not text:
                self.send_json(
                    {
                        "error": "Enter at least one word to visualise. "
                        "The input produced no word tokens."
                    },
                    400,
                )
                return
        else:
            text = DEFAULT_TEXT
        text_truncated = len(text) > MAX_TEXT_LENGTH
        text = text[:MAX_TEXT_LENGTH]

        def as_int(name, default, low, high):
            try:
                value = int(query.get(name, [default])[0])
            except (TypeError, ValueError):
                return default
            return max(low, min(high, value))

        embed_dim = as_int("embed_dim", 8, 2, 64)
        hidden_dim = as_int("hidden_dim", 32, 2, 256)
        max_seq_len = as_int("max_seq_len", 32, 4, 256)
        seed = as_int("seed", 0, 0, 10_000)

        try:
            payload = build_payload(
                text, embed_dim, hidden_dim, max_seq_len, seed
            )
        except Exception:
            traceback.print_exc()
            self.send_json(
                {"error": "Failed to run the transformer block."}, 500
            )
            return

        if not payload["tokens"]:
            self.send_json(
                {
                    "error": "No word tokens found in the input. "
                    "Enter at least one word."
                },
                400,
            )
            return

        payload["text_truncated"] = text_truncated
        self.send_json(payload, 200)

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
