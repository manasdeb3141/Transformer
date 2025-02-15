
import torch.nn as nn

from InputEmbedding import InputEmbedding
from PositionalEncoding import PositionalEncoding
from MultiheadAttention import MultiheadAttention
from FeedForward import FeedForward
from Encoder import EncoderSublayer, Encoder
from Decoder import DecoderSublayer, Decoder
from Projection import Projection

class Transformer(nn.Module):
    # Constructor
    def __init__(self, 
                 source_vocab_size: int, 
                 target_vocab_size: int, 
                 source_sequence_len: int,
                 target_sequence_len: int,
                 d_model: int = 512,
                 N_layers: int = 6,
                 N_heads: int = 8,
                 dropout: float = 0.1,
                 d_ff: int = 2048) -> None:
        super().__init__()

        #Create the embedding layers
        self._source_embed = InputEmbedding(d_model, source_vocab_size)
        self._target_embed = InputEmbedding(d_model, target_vocab_size)

        # Create the positional encoding layers
        self._source_pos_encoder = PositionalEncoding(d_model, source_sequence_len, dropout)
        self._target_pos_encoder = PositionalEncoding(d_model, target_sequence_len, dropout)

        # Create the Encoder sub-blocks
        encoder_sublayers = []
        for _ in range(N_layers):
            encoder_self_attention_block = MultiheadAttention(d_model, N_heads, dropout)
            feed_forward_block = FeedForward(d_model, d_ff, dropout)
            encoder_sub_block = EncoderSublayer(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_sublayers.append(encoder_sub_block)

        # Create the Decoder sub-blocks
        decoder_sublayers = []
        for _ in range(N_layers):
            decoder_self_attention_block = MultiheadAttention(d_model, N_heads, dropout)
            decoder_cross_attention_block = MultiheadAttention(d_model, N_heads, dropout)
            feed_forward_block = FeedForward(d_model, d_ff, dropout)
            decoder_sub_block = DecoderSublayer(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
            decoder_sublayers.append(decoder_sub_block)

        # Create the Encoder and Decoder
        self._encoder = Encoder(nn.ModuleList(encoder_sublayers))
        self._decoder = Decoder(nn.ModuleList(decoder_sublayers))

        # Create the Projection layer
        self._projection = Projection(d_model, target_vocab_size)

        # Initialize the parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, source, source_mask):
        source = self._source_embed(source)
        source = self._source_pos_encoder(source)
        return self._encoder(source, source_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self._target_embed(target)
        target = self._target_pos_encoder(target)
        return self._decoder(target, encoder_output, source_mask, target_mask)

    def project(self, x):
        return self._projection(x)


