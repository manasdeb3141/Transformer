import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_len: int, dropout: float) -> None:
        super().__init__()
        self._d_model = d_model
        self._sequence_len = sequence_len

        # To prevent overfitting while training
        self._dropout = nn.Dropout(dropout)

        # Create the positional encoding matrix of shape (sequence_len, d_model)
        position_enc = torch.zeros(sequence_len, d_model)

        # Vector of word positions of shape (sequence_len, 1)
        t = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1)

        # Compute omega using log for numerical stability
        omega = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encoding for the various word positions in a sequence
        position_enc[:, 0::2] = torch.sin(t * omega)
        position_enc[:, 1::2] = torch.cos(t * omega)

        # Add a singleton dimension to make the shape of the position encoding tensor (1, sequence_len, d_model)
        # We will be using batches of sentences and the first dimension corresponds to batches
        position_enc = position_enc.unsqueeze(0)

        # Register this as a model buffer so that no training is done on it
        # but it still shows up in the model's state_dict and is copied to the GPU
        self.register_buffer('_pos_enc', position_enc)


    def forward(self, x):
        #
        # Add the positional encoding vector to the input embedding vector
        # Note that x is the output of the InputEmbeddings layer and has 
        # the dimensions (batch_len, seq_len, d_model).  Therefore x.shape[1]
        # is equal to seq_len.
        #
        # Ignoring the batch dimension for a moment, each element of x along 
        # the seq_len (i.e. row) dimension is an InputEmbedding row vector of
        # length d_model. There are 'seq_len' number of rows in x and pos_enc
        # matrices.
        #
        # The following line adds a PositionalEncoding vector of length d_model
        # to the InputEmbedding vector. In other words, a specific row of the 
        # PositionalEncoding matrix gets added to the row of the matrix x, based 
        # on the row index. The row index is the position of the token in the 
        # sentence of length seq_len.
        #
        x = x + (self._pos_enc[:, :x.shape[1], :]).requires_grad_(False)
        return self._dropout(x)

