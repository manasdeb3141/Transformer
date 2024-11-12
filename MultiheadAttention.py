
import torch.nn as nn
import math

class MultiheadAttention(nn.Module):
    # Constructor
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self._d_model = d_model
        self._h = h
        assert d_model % h == 0, "Error: d_model must be divisible by h"

        self._d_k = d_model // h
        self._W_q = nn.Linear(d_model, d_model)
        self._W_k = nn.Linear(d_model, d_model)
        self._W_v = nn.Linear(d_model, d_model)

        self._W_o = nn.Linear(d_model, d_model)
        self._dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, sequence_len, d_k) -> (batch, h, sequence_len, sequence_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            #attention_scores.masked_fill_(mask == 0, -1e9)
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        # (batch, h, sequence_len, sequence_len)
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    
    def forward(self, q, k, v, mask):
        # (batch, sequence_len, d_model) -> (batch, sequence_len, d_model)
        query = self._W_q(q)

        # (batch, sequence_len, d_model) -> (batch, sequence_len, d_model)
        key = self._W_k(k)

        # (batch, sequence_len, d_model) -> (batch, sequence_len, d_model)
        value = self._W_v(v)


        # Split the query, key, value matrices into h parts along the embedding dimension

        # (batch, sequence_len, d_model) -> (batch, sequence_len, h, d_k) -> (batch, h, sequence_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self._h, self._d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self._h, self._d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self._h, self._d_k).transpose(1, 2)

        x, self._attention_scores = MultiheadAttention.attention(query, key, value, mask, self._dropout)

        # (batch, h, sequence_len, d_k) -> (batch, sequence_len, h, d_k) -> (batch, sequence_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self._h * self._d_k)

        # (batch, sequence_len, d_model) -> (batch, sequence_len, d_model)
        return self._W_o(x)
