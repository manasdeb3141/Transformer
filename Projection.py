
import torch
import torch.nn as nn


class Projection(nn.Module):
    # Constructor
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self._proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, sequence_len, d_model) -> (batch, sequence_len, vocab_size)
        return torch.log_softmax(self._proj(x), dim = -1)