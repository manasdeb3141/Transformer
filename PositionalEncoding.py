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
        # Add the positional encoding vector to the input embedding vector
        x = x + (self._pos_enc[:, :x.shape[1], :]).requires_grad_(False)
        return self._dropout(x)

