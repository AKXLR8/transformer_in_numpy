import re
import numpy as np

class Tokenizers:
    def __init__(self,text):
        words = re.findall(r"\b\w+\b",text.lower())
        self.vocab = ["<UNK>"] + sorted(set(words))
        self.word_to_id = {
            word: index
            for index,word in enumerate(self.vocab)
        }

        self.id_to_word = {
            index:word
            for index,word in enumerate(self.vocab)
        }

    def encode(self,text):
        words = re.findall(r"\b\w+\b",text.lower())
        return [
            self.word_to_id.get(word,0)
            for word in words 
        ]
    def decode(self,token_ids):
        return " ".join(
            self.id_to_word.get(token_id,"<UNK>")
            for token_id in token_ids
        )


class Tokenembedding:
    def __init__(self,vocab_size,embed_dim):
        self.weights = np.random.randn(
            vocab_size,embed_dim
        ) * 0.02

    def forward(self,token_ids):
        return self.weights[token_ids]


class Positionembedding:
    def __init__(self,maxseq_len,embeddding_dim):
        self.weights = np.random.randn(maxseq_len,embeddding_dim) * 0.02

    def forward(self,token_embeddings):
        seqlen = token_embeddings.shape[0]
        positions = self.weights[:seqlen]
        return token_embeddings + positions


def softmax(x, axis = -1):
    X = x - np.max(x,axis = axis, keepdims = True)
    exp_x = np.exp(X)
    return exp_x/np.sum(
        exp_x,
        axis = axis,
        keepdims = True
    )


class SelfAttention:
    def __init__(self,embedding_dim):
        self.Wq = np.random.randn(embedding_dim , embedding_dim)*0.02
        self.Wk = (
            np.random.randn(embedding_dim, embedding_dim)
            * 0.02
        )
        self.Wv = (
            np.random.randn(embedding_dim, embedding_dim)
            * 0.02
        )
        self.Wo = (
            np.random.randn(embedding_dim, embedding_dim)
            * 0.02
        )
        self.attention_weights = None
        self.scores = None

    def forward(self,x):
        seq_len,embedding_dim = x.shape

        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        scores = Q @ K.T
        scores /= np.sqrt(embedding_dim)
        mask = np.triu(
            np.ones((seq_len, seq_len), dtype=bool),
            k = 1
        )
        scores[mask] = -np.inf
        attention_weights = softmax(scores,axis =-1)
        context = attention_weights @ V
        output = context @ self.Wo

        self.attention_weights = attention_weights
        self.scores = np.where(
            np.isfinite(scores), scores, 0.0
        )

        return output


#GELU activation func
def gelu(x):
    return 0.5 * x * (
        1.0 + np.tanh(
            np.sqrt(2.0 / np.pi)
            * (x + 0.044715 * x**3)
        )
    )


class FeedForward:
    def __init__(self, embedding_dim, hidden_dim):
        self.W1 = (
            np.random.randn(embedding_dim, hidden_dim)
            * 0.02
        )

        self.b1 = np.zeros(hidden_dim)

        self.W2 = (
            np.random.randn(hidden_dim, embedding_dim)
            * 0.02
        )

        self.b2 = np.zeros(embedding_dim)

    def forward(self, x):
        hidden = x @ self.W1 + self.b1

        hidden = gelu(hidden)

        output = hidden @ self.W2 + self.b2

        return output


#layernormlization
class LayerNorm:
    def __init__(self, embedding_dim, eps=1e-5):
        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)

        variance = np.mean(
            (x - mean) ** 2,
            axis=-1,
            keepdims=True
        )

        normalized = (
            (x - mean)
            / np.sqrt(variance + self.eps)
        )

        return self.gamma * normalized + self.beta


class TransformerPipeline:
    """Runs the full block end to end and keeps every intermediate tensor.

    Used by the visualizer front end so the UI reflects the real numpy
    implementation rather than a re-implementation.
    """

    def __init__(
        self,
        text,
        embed_dim=8,
        hidden_dim=32,
        max_seq_len=32,
        seed=0,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.text = text
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.tokenizer = Tokenizers(text)
        token_ids = self.tokenizer.encode(text)
        self.truncated = len(token_ids) > max_seq_len
        self.token_ids = token_ids[:max_seq_len]
        self.tokens = [
            self.tokenizer.id_to_word.get(token_id, "<UNK>")
            for token_id in self.token_ids
        ]

        self.token_embedding = Tokenembedding(
            vocab_size=len(self.tokenizer.vocab),
            embed_dim=embed_dim,
        )
        self.position_embedding = Positionembedding(
            maxseq_len=max_seq_len,
            embeddding_dim=embed_dim,
        )
        self.attention = SelfAttention(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.layernorm = LayerNorm(embed_dim)

    def run(self):
        if not self.token_ids:
            return self._empty_payload()

        token_ids = np.array(self.token_ids, dtype=int)

        token_vectors = self.token_embedding.forward(token_ids)
        positions = self.position_embedding.weights[: len(token_ids)]
        x = token_vectors + positions

        attention_output = self.attention.forward(x)
        residual_1 = x + attention_output
        norm_1 = self.layernorm.forward(residual_1)

        ffn_hidden = gelu(norm_1 @ self.ffn.W1 + self.ffn.b1)
        ffn_output = self.ffn.forward(norm_1)
        residual_2 = norm_1 + ffn_output
        norm_2 = self.layernorm.forward(residual_2)

        return {
            "text": self.text,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "vocab": self.tokenizer.vocab,
            "vocab_size": len(self.tokenizer.vocab),
            "truncated": self.truncated,
            "config": {
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "max_seq_len": self.max_seq_len,
                "seq_len": len(self.token_ids),
            },
            "token_embedding": token_vectors.tolist(),
            "position_embedding": positions.tolist(),
            "attention_weights": self.attention.attention_weights.tolist(),
            "attention_scores": self.attention.scores.tolist(),
            "attention_output": attention_output.tolist(),
            "residual_1": residual_1.tolist(),
            "norm_1": norm_1.tolist(),
            "ffn_hidden": ffn_hidden.tolist(),
            "ffn_output": ffn_output.tolist(),
            "residual_2": residual_2.tolist(),
            "norm_2": norm_2.tolist(),
            "shapes": {
                "token_ids": list(token_ids.shape),
                "token_embedding": list(token_vectors.shape),
                "position_embedding": list(positions.shape),
                "combined": list(x.shape),
                "attention_output": list(attention_output.shape),
                "residual_1": list(residual_1.shape),
                "norm_1": list(norm_1.shape),
                "ffn_hidden": list(ffn_hidden.shape),
                "ffn_output": list(ffn_output.shape),
                "residual_2": list(residual_2.shape),
                "norm_2": list(norm_2.shape),
            },
            "params": self.param_count(),
        }

    def _empty_payload(self):
        """Shape-correct payload for input that contains no word tokens."""
        d = self.embed_dim
        h = self.hidden_dim
        return {
            "text": self.text,
            "tokens": [],
            "token_ids": [],
            "vocab": self.tokenizer.vocab,
            "vocab_size": len(self.tokenizer.vocab),
            "truncated": False,
            "config": {
                "embed_dim": d,
                "hidden_dim": h,
                "max_seq_len": self.max_seq_len,
                "seq_len": 0,
            },
            "token_embedding": [],
            "position_embedding": [],
            "attention_weights": [],
            "attention_scores": [],
            "attention_output": [],
            "residual_1": [],
            "norm_1": [],
            "ffn_hidden": [],
            "ffn_output": [],
            "residual_2": [],
            "norm_2": [],
            "shapes": {
                "token_ids": [0],
                "token_embedding": [0, d],
                "position_embedding": [0, d],
                "combined": [0, d],
                "attention_output": [0, d],
                "residual_1": [0, d],
                "norm_1": [0, d],
                "ffn_hidden": [0, h],
                "ffn_output": [0, d],
                "residual_2": [0, d],
                "norm_2": [0, d],
            },
            "params": self.param_count(),
        }

    def param_count(self):
        return {
            "token_embedding": int(self.token_embedding.weights.size),
            "position_embedding": int(self.position_embedding.weights.size),
            "attention": int(
                self.attention.Wq.size
                + self.attention.Wk.size
                + self.attention.Wv.size
                + self.attention.Wo.size
            ),
            "feed_forward": int(
                self.ffn.W1.size
                + self.ffn.b1.size
                + self.ffn.W2.size
                + self.ffn.b2.size
            ),
            "layer_norm": int(
                self.layernorm.gamma.size + self.layernorm.beta.size
            ),
            "total": int(
                self.token_embedding.weights.size
                + self.position_embedding.weights.size
                + self.attention.Wq.size
                + self.attention.Wk.size
                + self.attention.Wv.size
                + self.attention.Wo.size
                + self.ffn.W1.size
                + self.ffn.b1.size
                + self.ffn.W2.size
                + self.ffn.b2.size
                + self.layernorm.gamma.size
                + self.layernorm.beta.size
            ),
        }


def main():
    text = "akshay is running the dog is barking"

    tokenizer = Tokenizers(text)

    encoded = tokenizer.encode("akshay is running")
    print("Vocabulary:", tokenizer.vocab)
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))

    embedding = Tokenembedding(
        vocab_size = len(tokenizer.vocab),
        embed_dim = 8
    )

    token_ids = np.array(
        tokenizer.encode("the dog is barking")
    )

    vector = embedding.forward(token_ids)

    print("Token ID shape ; ",token_ids.shape)
    print("Embedding shape:", vector.shape)
    print(vector)

    position_embedding = Positionembedding(
        maxseq_len=32,
        embeddding_dim=8
    )
    x = position_embedding.forward(vector)
    print("combined embeddng shape:",x.shape)

    attention = SelfAttention(embedding_dim=8)

    attention_output = attention.forward(x)

    print("Attention output shape:", attention_output.shape)

    ffn = FeedForward(
        embedding_dim=8,
        hidden_dim=32
    )

    ffn_output = ffn.forward(attention_output)

    print("Feed-forward output:", ffn_output.shape)

    sublayer_output = attention.forward(x)

    output = x + sublayer_output

    print("Input shape:", x.shape)
    print("Sublayer output shape:", sublayer_output.shape)
    print("Residual output shape:", output.shape)


if __name__ == "__main__":
    main()
