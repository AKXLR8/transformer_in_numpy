"""Correctness tests for the transformer block and the visualizer contract.

    python -m unittest discover -s tests -v

Covers the math invariants of the block, the edge cases the front end can
send, and the JSON contract that the UI and the Netlify function rely on.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import (  # noqa: E402
    FeedForward,
    LayerNorm,
    SelfAttention,
    Tokenizers,
    TransformerPipeline,
    gelu,
)

TEXT = "akshay is running the dog is barking"


class TestMath(unittest.TestCase):
    def setUp(self):
        self.pipeline = TransformerPipeline(TEXT, embed_dim=8, hidden_dim=32, seed=0)
        self.payload = self.pipeline.run()
        p = self.payload
        self.W = np.array(p["attention_weights"])
        self.tok = np.array(p["token_embedding"])
        self.pos = np.array(p["position_embedding"])
        self.attn_out = np.array(p["attention_output"])
        self.res1 = np.array(p["residual_1"])
        self.norm1 = np.array(p["norm_1"])
        self.ffn_hidden = np.array(p["ffn_hidden"])
        self.ffn_out = np.array(p["ffn_output"])
        self.res2 = np.array(p["residual_2"])
        self.norm2 = np.array(p["norm_2"])

    def test_attention_rows_sum_to_one(self):
        np.testing.assert_allclose(self.W.sum(axis=1), 1.0, atol=1e-9)

    def test_attention_is_non_negative(self):
        self.assertTrue((self.W >= 0).all())

    def test_causal_mask_zeroes_upper_triangle(self):
        np.testing.assert_allclose(np.triu(self.W, 1), 0.0)

    def test_residual_adds_attention_to_input(self):
        np.testing.assert_allclose(self.res1, self.tok + self.pos + self.attn_out, atol=1e-9)

    def test_layer_norm_matches_reference(self):
        ln = LayerNorm(8)
        np.testing.assert_allclose(self.norm1, ln.forward(self.res1), atol=1e-9)
        np.testing.assert_allclose(self.norm2, ln.forward(self.res2), atol=1e-9)

    def test_ffn_hidden_is_gelu_of_first_layer(self):
        np.testing.assert_allclose(
            self.ffn_hidden,
            gelu(self.norm1 @ self.pipeline.ffn.W1 + self.pipeline.ffn.b1),
            atol=1e-9,
        )

    def test_ffn_output_matches_forward(self):
        np.testing.assert_allclose(
            self.ffn_out, self.pipeline.ffn.forward(self.norm1), atol=1e-9
        )

    def test_second_residual(self):
        np.testing.assert_allclose(self.res2, self.norm1 + self.ffn_out, atol=1e-9)

    def test_final_layer_norm_is_standardised(self):
        np.testing.assert_allclose(self.norm2.mean(axis=1), 0.0, atol=1e-8)
        np.testing.assert_allclose(self.norm2.var(axis=1), 1.0, atol=1e-4)

    def test_softmax_is_numerically_stable(self):
        attention = SelfAttention(4)
        attention.Wq[:] = 1e3
        attention.Wk[:] = 1e3
        attention.forward(np.ones((3, 4)) * 100)
        self.assertTrue(np.isfinite(attention.attention_weights).all())


class TestContract(unittest.TestCase):
    def test_shapes_match_config(self):
        p = TransformerPipeline(TEXT, embed_dim=6, hidden_dim=16, seed=0).run()
        seq = p["config"]["seq_len"]
        self.assertEqual(p["shapes"]["attention_output"], [seq, 6])
        self.assertEqual(p["shapes"]["ffn_hidden"], [seq, 16])
        self.assertEqual(len(p["tokens"]), len(p["token_ids"]))
        self.assertEqual(p["vocab_size"], len(p["vocab"]))
        self.assertEqual(p["vocab"][0], "<UNK>")

    def test_param_counts_add_up(self):
        p = TransformerPipeline(TEXT, seed=0).run()
        params = p["params"]
        self.assertEqual(
            params["total"],
            params["token_embedding"]
            + params["position_embedding"]
            + params["attention"]
            + params["feed_forward"]
            + params["layer_norm"],
        )

    def test_determinism(self):
        a = TransformerPipeline(TEXT, seed=7).run()
        b = TransformerPipeline(TEXT, seed=7).run()
        c = TransformerPipeline(TEXT, seed=8).run()
        self.assertEqual(a["attention_weights"], b["attention_weights"])
        self.assertNotEqual(a["attention_weights"], c["attention_weights"])


class TestEdgeCases(unittest.TestCase):
    def test_no_tokens_returns_empty_payload(self):
        for text in ("", "   ", "!!!", "???"):
            p = TransformerPipeline(text, seed=0).run()
            self.assertEqual(p["tokens"], [])
            self.assertEqual(p["shapes"]["token_embedding"], [0, 8])
            self.assertEqual(p["shapes"]["attention_output"], [0, 8])

    def test_single_token(self):
        p = TransformerPipeline("hello", seed=0).run()
        self.assertEqual(p["tokens"], ["hello"])
        self.assertEqual(p["shapes"]["attention_output"], [1, 8])
        np.testing.assert_allclose(np.array(p["attention_weights"]), [[1.0]], atol=1e-9)

    def test_repeated_words_share_one_id(self):
        p = TransformerPipeline("the the the", seed=0).run()
        self.assertEqual(len(set(p["token_ids"])), 1)

    def test_sequence_is_truncated_to_max_len(self):
        text = " ".join(f"word{i}" for i in range(200))
        p = TransformerPipeline(text, max_seq_len=32, seed=0).run()
        self.assertEqual(p["config"]["seq_len"], 32)
        self.assertTrue(p["truncated"])

    def test_unicode_and_emoji(self):
        p = TransformerPipeline("café 你好 👋", seed=0).run()
        self.assertTrue(p["tokens"])
        self.assertTrue(all(t != "<UNK>" for t in p["tokens"]))

    def test_extreme_dimensions(self):
        for embed_dim, hidden_dim in ((2, 2), (2, 256), (64, 2)):
            p = TransformerPipeline(TEXT, embed_dim=embed_dim,
                                    hidden_dim=hidden_dim, seed=0).run()
            W = np.array(p["attention_weights"])
            np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-9)


class TestTokenizer(unittest.TestCase):
    def test_regex_splits_on_word_boundaries(self):
        tok = Tokenizers("hello world")
        self.assertEqual(tok.encode("hello world"), [1, 2])

    def test_punctuation_is_dropped(self):
        tok = Tokenizers("hello, world!")
        self.assertEqual(tok.decode(tok.encode("hello, world!")), "hello world")

    def test_unknown_words_map_to_unk(self):
        tok = Tokenizers("hello")
        self.assertEqual(tok.encode("zebra")[0], 0)

    def test_lowercases(self):
        tok = Tokenizers("Hello World")
        self.assertEqual(tok.encode("HELLO"), tok.encode("hello"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
