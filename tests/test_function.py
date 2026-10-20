"""Tests for the Netlify function - the production entry point.

The UI reads a fixed set of keys from the JSON payload, so these tests
pin down that contract as well as the function's status codes.
"""

import importlib.util
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FUNCTION_PATH = os.path.join(ROOT, "netlify", "functions", "run.py")

_spec = importlib.util.spec_from_file_location("netlify_run", FUNCTION_PATH)
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


def call(**params):
    event = {"queryStringParameters": params}
    response = run_fn.handler(event, None)
    return response["statusCode"], json.loads(response["body"])


class TestFunction(unittest.TestCase):
    def test_returns_200_and_the_full_payload(self):
        status, body = call(text="the cat sat on the mat")
        self.assertEqual(status, 200)
        for key in UI_KEYS:
            self.assertIn(key, body, f"UI expects '{key}' in the response")
        self.assertEqual(body["tokens"], ["the", "cat", "sat", "on", "the", "mat"])

    def test_config_reflects_the_parameters(self):
        status, body = call(text="hello world", embed_dim="6", hidden_dim="16")
        self.assertEqual(status, 200)
        self.assertEqual(body["config"]["embed_dim"], 6)
        self.assertEqual(body["config"]["hidden_dim"], 16)
        self.assertEqual(body["shapes"]["attention_output"], [2, 6])
        self.assertEqual(body["shapes"]["ffn_hidden"], [2, 16])

    def test_missing_text_falls_back_to_the_default_sentence(self):
        status, body = call()
        self.assertEqual(status, 200)
        self.assertTrue(body["tokens"])

    def test_blank_text_is_rejected(self):
        status, body = call(text="   ")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_text_without_word_tokens_is_rejected(self):
        status, body = call(text="!!! ???")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_long_text_is_capped_and_flagged(self):
        status, body = call(text="word " * 300)
        self.assertEqual(status, 200)
        self.assertTrue(body["text_truncated"])
        self.assertLessEqual(len(body["text"]), 500)

    def test_short_text_is_not_flagged_as_truncated(self):
        status, body = call(text="hello")
        self.assertEqual(status, 200)
        self.assertFalse(body["text_truncated"])

    def test_out_of_range_parameters_are_clamped(self):
        status, body = call(text="hello world", embed_dim="9999", hidden_dim="-5")
        self.assertEqual(status, 200)
        self.assertLessEqual(body["config"]["embed_dim"], 64)
        self.assertGreaterEqual(body["config"]["hidden_dim"], 2)

    def test_garbage_parameters_fall_back_to_defaults(self):
        status, body = call(text="hello", embed_dim="abc", seed="xyz")
        self.assertEqual(status, 200)
        self.assertEqual(body["config"]["embed_dim"], 8)

    def test_same_seed_gives_the_same_weights(self):
        _, first = call(text="hello world", seed="42")
        _, second = call(text="hello world", seed="42")
        self.assertEqual(first["attention_weights"], second["attention_weights"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
