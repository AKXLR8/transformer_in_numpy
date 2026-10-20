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

text = "akshay is running the dog is barking"

tokenizer = Tokenizers(text)

encoded = tokenizer.encode("akshay is running")
print("Vocabulary:", tokenizer.vocab)
print("Encoded:", encoded)
print("Decoded:", tokenizer.decode(encoded))

class Tokenembedding:
    def __init__(self,vocab_size,embed_dim):
        self.weights = np.random.randn(
            vocab_size,embed_dim
        ) * 0.02

    def forward(self,token_ids):
        return self.weights[token_ids]

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

class Positionembedding:
    def __init__(self,maxseq_len,embeddding_dim):
        self.weights = np.random.randn(maxseq_len,embeddding_dim) * 0.02

    def forward(self,token_embeddings):
        seqlen = token_embeddings.shape[0]
        positions = self.weights[:seqlen]
        return token_embeddings + positions

position_embedding = Positionembedding(
    maxseq_len=32,
    embeddding_dim=8
)
x = position_embedding.forward(vector)
print("combined embeddng shape:",x.shape)

def softmax(x, axis = -1):
    X = x - np.max(x,axis = axis, keepdims = True)
    exp_x = np.exp(x)
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

        return output

attention = SelfAttention(embedding_dim=8)

attention_output = attention.forward(x)

print("Attention output shape:", attention_output.shape)

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

ffn = FeedForward(
    embedding_dim=8,
    hidden_dim=32
)

ffn_output = ffn.forward(attention_output)

print("Feed-forward output:", ffn_output.shape)


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

sublayer_output = attention.forward(x)

output = x + sublayer_output

print("Input shape:", x.shape)
print("Sublayer output shape:", sublayer_output.shape)
print("Residual output shape:", output.shape)
