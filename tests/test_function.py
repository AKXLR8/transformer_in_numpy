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
    """The class Vercel loads: api/run.py -> handler(BaseHTTPRequestHandler)."""

    def test_handler_class_exists_and_has_the_right_shape(self):
        self.assertTrue(hasattr(run_fn, "handler"))
        self.assertTrue(hasattr(run_fn.handler, "do_GET"))
        self.assertEqual(
            run_fn.handler.__mro__[1].__name__, "BaseHTTPRequestHandler"
        )

    def test_handler_serves_the_api_route(self):
        import io
        from http.server import BaseHTTPRequestHandler

        captured = {}

        class FakeSocket:
            def makefile(self, *args, **kwargs):
                return io.BytesIO()

        original = run_fn.handler.log_message
        run_fn.handler.log_message = lambda *a, **k: None
        try:
            request = run_fn.handler.__new__(run_fn.handler)
            request.path = "/api/run?text=hello%20world"
            request.command = "GET"
            request.requestline = "GET /api/run?text=hello%20world"
            request.client_address = ("127.0.0.1", 1234)
            request.server = None
            request.close_connection = True
            request.wfile = io.BytesIO()

            def send_response(code, message=None):
                captured["status"] = code

            def send_header(key, value):
                captured.setdefault("headers", {})[key] = value

            def end_headers():
                captured["ended"] = True

            request.send_response = send_response
            request.send_header = send_header
            request.end_headers = end_headers
            request.send_error = lambda *a, **k: captured.update(status=404)

            run_fn.handler.do_GET(request)
        finally:
            run_fn.handler.log_message = original

        self.assertEqual(captured.get("status"), 200)
        body = json.loads(request.wfile.getvalue().decode("utf-8"))
        self.assertEqual(body["tokens"], ["hello", "world"])
        self.assertEqual(captured["headers"]["Content-Type"], "application/json")
        del BaseHTTPRequestHandler


if __name__ == "__main__":
    unittest.main(verbosity=2)
