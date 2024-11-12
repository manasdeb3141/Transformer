
import torch.nn as nn

from MultiheadAttention import MultiheadAttention
from FeedForward import FeedForward
from ResidualConnection import ResidualConnection
from LayerNormalization import LayerNormalization

class EncoderSublayer(nn.Module):
    # Constructor
    def __init__(self, self_attention : MultiheadAttention, feed_forward : FeedForward, dropout:float) -> None:
        super().__init__()
        self._self_attention = self_attention
        self._feed_forward = feed_forward
        self._residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self._residual_connections[0](x, lambda x: self._self_attention(x, x, x, src_mask))
        x = self._residual_connections[1](x, self._feed_forward)
        return x

    
class Encoder(nn.Module):
    # Constructor
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self._layers = layers
        self._norm = LayerNormalization()


    def forward(self, x, mask):
        for layer in self._layers:
            x = layer(x, mask)

        return self._norm(x)