"""Shared request handling for the visualizer API.

Both entry points use this module so they can never drift apart:
  * server.py     - local development server
  * api/run.py    - Vercel function (serves /api/run)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer import TransformerPipeline  # noqa: E402

DEFAULT_TEXT = "akshay is running the dog is barking"
MAX_TEXT_LENGTH = 500


def as_int(params, name, default, low, high):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def build_payload(params):
    """Build the full JSON payload for one request.

    `params` is a mapping of query string values. Raises ValueError with a
    message that is safe to show the user.
    """
    raw_text = params.get("text")
    if raw_text is None:
        text = DEFAULT_TEXT
    else:
        text = str(raw_text).strip()
        if not text:
            raise ValueError(
                "Enter at least one word to visualise. "
                "The input produced no word tokens."
            )

    text_truncated = len(text) > MAX_TEXT_LENGTH
    text = text[:MAX_TEXT_LENGTH]

    payload = TransformerPipeline(
        text=text,
        embed_dim=as_int(params, "embed_dim", 8, 2, 64),
        hidden_dim=as_int(params, "hidden_dim", 32, 2, 256),
        max_seq_len=as_int(params, "max_seq_len", 32, 4, 256),
        seed=as_int(params, "seed", 0, 0, 10000),
    ).run()

    if not payload["tokens"]:
        raise ValueError(
            "No word tokens found in the input. Enter at least one word."
        )

    payload["text_truncated"] = text_truncated
    return payload


def handle_params(params):
    """Return (status_code, body) for one request."""
    try:
        return 200, build_payload(params)
    except ValueError as error:
        return 400, {"error": str(error)}
    except Exception as error:  # pragma: no cover
        return 500, {"error": f"Failed to run the transformer block: {error}"}
