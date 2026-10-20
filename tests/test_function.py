"""Tests for the deployed API - the Vercel function and the shared module.

The UI reads a fixed set of keys from the JSON payload, so these tests pin
down that contract as well as the status codes.
"""

import importlib.util
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformer_api import build_payload, handle_params  # noqa: E402

FUNCTION_PATH = os.path.join(ROOT, "api", "run.py")
_spec = importlib.util.spec_from_file_location("vercel_run", FUNCTION_PATH)
run_fn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_fn)

# everything static/app.js reads out of the response
UI_KEYS = [
    "tokens",
    "token_ids",
    "vocab",
    "vocab_size",
    "text",
    "config",
    "token_embedding",
    "position_embedding",
    "attention_weights",
    "attention_scores",
    "attention_output",
    "residual_1",
    "norm_1",
    "ffn_hidden",
    "ffn_output",
    "residual_2",
    "norm_2",
    "shapes",
    "params",
    "text_truncated",
]


class TestFunction(unittest.TestCase):
    def test_returns_200_and_the_full_payload(self):
        status, body = handle_params({"text": "the cat sat on the mat"})
        self.assertEqual(status, 200)
        for key in UI_KEYS:
            self.assertIn(key, body, f"UI expects '{key}' in the response")
        self.assertEqual(body["tokens"], ["the", "cat", "sat", "on", "the", "mat"])

    def test_config_reflects_the_parameters(self):
        status, body = handle_params(
            {"text": "hello world", "embed_dim": "6", "hidden_dim": "16"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["config"]["embed_dim"], 6)
        self.assertEqual(body["config"]["hidden_dim"], 16)
        self.assertEqual(body["shapes"]["attention_output"], [2, 6])
        self.assertEqual(body["shapes"]["ffn_hidden"], [2, 16])

    def test_missing_text_falls_back_to_the_default_sentence(self):
        status, body = handle_params({})
        self.assertEqual(status, 200)
        self.assertTrue(body["tokens"])

    def test_blank_text_is_rejected(self):
        status, body = handle_params({"text": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_empty_query_string_value_is_rejected(self):
        # parse_qs drops blank values unless asked not to, so make sure the
        # entry points keep them: an emptied text box must not silently fall
        # back to the default sentence
        from urllib.parse import parse_qs

        for keep in (False, True):
            parsed = parse_qs("text=", keep_blank_values=keep)
            self.assertEqual(
                handle_params({k: v[0] for k, v in parsed.items()})[0],
                400 if keep else 200,
            )

    def test_text_without_word_tokens_is_rejected(self):
        status, body = handle_params({"text": "!!! ???"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_long_text_is_capped_and_flagged(self):
        status, body = handle_params({"text": "word " * 300})
        self.assertEqual(status, 200)
        self.assertTrue(body["text_truncated"])
        self.assertLessEqual(len(body["text"]), 500)

    def test_short_text_is_not_flagged_as_truncated(self):
        status, body = handle_params({"text": "hello"})
        self.assertEqual(status, 200)
        self.assertFalse(body["text_truncated"])

    def test_out_of_range_parameters_are_clamped(self):
        status, body = handle_params(
            {"text": "hello world", "embed_dim": "9999", "hidden_dim": "-5"}
        )
        self.assertEqual(status, 200)
        self.assertLessEqual(body["config"]["embed_dim"], 64)
        self.assertGreaterEqual(body["config"]["hidden_dim"], 2)

    def test_garbage_parameters_fall_back_to_defaults(self):
        status, body = handle_params(
            {"text": "hello", "embed_dim": "abc", "seed": "xyz"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["config"]["embed_dim"], 8)

    def test_same_seed_gives_the_same_weights(self):
        _, first = handle_params({"text": "hello world", "seed": "42"})
        _, second = handle_params({"text": "hello world", "seed": "42"})
        self.assertEqual(first["attention_weights"], second["attention_weights"])


class TestVercelHandler(unittest.TestCase):
    """The class Vercel loads: api/run.py -> handler(BaseHTTPRequestHandler).

    Vercel routes every request to this one function, so it has to serve the
    API *and* the static site.
    """

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer

        cls.original_log = run_fn.handler.log_message
        run_fn.handler.log_message = lambda *a, **k: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), run_fn.handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        run_fn.handler.log_message = cls.original_log

    def get(self, path):
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read()

    def test_handler_class_exists_and_has_the_right_shape(self):
        self.assertTrue(hasattr(run_fn, "handler"))
        self.assertTrue(hasattr(run_fn.handler, "do_GET"))
        self.assertEqual(
            run_fn.handler.__mro__[1].__name__, "BaseHTTPRequestHandler"
        )

    def test_api_route_returns_json(self):
        status, content_type, body = self.get("/api/run?text=hello%20world")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(json.loads(body)["tokens"], ["hello", "world"])

    def test_root_serves_the_site(self):
        status, content_type, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"<title>", body)

    def test_static_assets_are_served_with_the_right_types(self):
        status, content_type, body = self.get("/app.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", content_type)
        self.assertGreater(len(body), 1000)

        status, content_type, body = self.get("/styles.css")
        self.assertEqual(status, 200)
        self.assertIn("text/css", content_type)

    def test_missing_file_is_404(self):
        status, _, _ = self.get("/nope.txt")
        self.assertEqual(status, 404)

    def test_path_traversal_is_blocked(self):
        status, _, _ = self.get("/../transformer.py")
        self.assertEqual(status, 404)

    def test_root_with_api_parameters_returns_json(self):
        # the front end always sends the tuning parameters, so a request that
        # lands on "/" with them is an API call
        status, content_type, body = self.get(
            "/?text=hello&embed_dim=8&hidden_dim=32&max_seq_len=32&seed=0"
        )
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(json.loads(body)["tokens"], ["hello"])

    def test_root_with_only_text_serves_the_page(self):
        # the shareable url format is /?text=... - that must stay a page
        status, content_type, _ = self.get("/?text=hello")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)


if __name__ == "__main__":
    unittest.main(verbosity=2)
