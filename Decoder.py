
import torch.nn as nn

from MultiheadAttention import MultiheadAttention
from FeedForward import FeedForward
from ResidualConnection import ResidualConnection
from LayerNormalization import LayerNormalization


class DecoderSublayer(nn.Module):
    # Constructor
    def __init__(self, self_attention : MultiheadAttention, cross_attention: MultiheadAttention, feed_forward : FeedForward, dropout:float) -> None:
        super().__init__()
        self._self_attention = self_attention
        self._cross_attention = cross_attention
        self._feed_forward = feed_forward
        self._residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self._residual_connections[0](x, lambda x: self._self_attention(x, x, x, target_mask))
        x = self._residual_connections[1](x, lambda x: self._cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self._residual_connections[2](x, self._feed_forward)
        return x


class Decoder(nn.Module):
    # Constructor
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self._layers = layers
        self._norm = LayerNormalization()


    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self._layers:
            x = layer(x, encoder_output, src_mask, target_mask)

        return self._norm(x)
