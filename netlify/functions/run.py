"""Netlify Function that runs the transformer block and returns JSON.

Deployed at /.netlify/functions/run.
The static front end calls /api/run, which netlify.toml rewrites here.
"""

import json
import os
import sys

DEFAULT_TEXT = "akshay is running the dog is barking"
MAX_TEXT_LENGTH = 500


def _find_root():
    """Locate the directory that holds transformer.py.

    Netlify bundles the function with the repository files, but the exact
    nesting inside the bundle can vary, so probe a few candidate roots.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        here,
        os.path.dirname(here),
        os.path.dirname(os.path.dirname(here)),
        os.path.dirname(os.path.dirname(os.path.dirname(here))),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "transformer.py")):
            return candidate
    return here


ROOT = _find_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from transformer import TransformerPipeline
except ImportError:  # pragma: no cover
    print("transformer.py not found next to the function bundle", file=sys.stderr)
    raise


def _as_int(params, name, default, low, high):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def build_payload(params):
    text = (params.get("text") or "").strip() or DEFAULT_TEXT
    text = text[:MAX_TEXT_LENGTH]

    embed_dim = _as_int(params, "embed_dim", 8, 2, 64)
    hidden_dim = _as_int(params, "hidden_dim", 32, 2, 256)
    max_seq_len = _as_int(params, "max_seq_len", 32, 4, 256)
    seed = _as_int(params, "seed", 0, 0, 10000)

    pipeline = TransformerPipeline(
        text=text,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    return pipeline.run()


def handler(event, context):
    params = event.get("queryStringParameters") or {}

    try:
        payload = build_payload(params)
    except Exception as error:  # pragma: no cover
        return _json({"error": f"Failed to run the transformer block: {error}"}, 500)

    if not payload["tokens"]:
        return _json(
            {"error": "No word tokens found in the input. Enter at least one word."},
            400,
        )

    return _json(payload, 200)


def _json(data, status=200):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
        },
        "body": json.dumps(data),
    }
