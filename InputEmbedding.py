import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    # Constructor
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self._d_model = d_model
        self._vocab_size = vocab_size

        # Use the PyTorch Embedding layer to learn the input word emebeddings
        # during training
        self._embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Scale the input embeddings by sqrt(d_model) as described in
        # the paper
        return self._embedding(x) * math.sqrt(self._d_model)