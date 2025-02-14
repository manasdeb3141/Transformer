import torch
from torch_mist.estimators import mine
from torch_mist.utils.train import train_mi_estimator
from torch_mist.utils import evaluate_mi
import numpy as np

class MINE_estimator:
    def __init__(self, X, Y) -> None:
        super().__init__()
        self._X = torch.from_numpy(X.astype(np.float32))
        self._Y = torch.from_numpy(Y.astype(np.float32))

    def run(self) -> float:

        # Instantiate the JS mutual information estimator
        # estimator = mine(x_dim=1, y_dim=1, hidden_dims=[64, 32])
        estimator = mine(x_dim=1, y_dim=1, hidden_dims=[256, 128, 128, 64])

        # Train it on the given samples
        train_log = train_mi_estimator(
            estimator=estimator,
            train_data=(self._X, self._Y),
            batch_size=64,
            max_iterations=1000,
            device='cuda'
        )

        # Evaluate the estimator on the entirety of the data
        estimated_mi = evaluate_mi(estimator=estimator, data=(self._X, self._Y), batch_size=64, device='cuda')

        return estimated_mi, train_log
