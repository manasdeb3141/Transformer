import torch
import torch.nn as nn

class FeedForward(nn.Module):
    # Constructor
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        # W1 and B1
        self._linear_1 = nn.Linear(d_model, d_ff)

        self._dropout = nn.Dropout(dropout)

        # W2 and B2
        self._linear_2 = nn.Linear(d_ff, d_model)


    def forward(self, x):
        # (batch, sequence_len, d_model) -> (batch, sequence_len, d_ff) -> (batch, sequence_len, d_model)
        return self._linear_2(self._dropout(torch.relu(self._linear_1(x))))


