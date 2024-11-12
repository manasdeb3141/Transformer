
import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    # Constructor
    def __init__(self, eps_val: float = 10**-6) -> None:
        super(LayerNormalization, self).__init__()
        self._eps_val = eps_val

        # Make these parameters learnable
        self._alpha = nn.Parameter(torch.ones(1))
        self._bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean_val = x.mean(dim = -1, keepdim = True)
        std_val = x.std(dim = -1, keepdim = True)
        x_hat = (x - mean_val) / (std_val + self._eps_val)
        return (self._alpha * x_hat) + self._bias
    
