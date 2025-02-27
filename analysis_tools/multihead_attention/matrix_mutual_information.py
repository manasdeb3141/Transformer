
import sys
sys.path.append('../utils')

import numpy as np
from tqdm import tqdm
from mutual_info_estimator import MutualInfoEstimator

def compute_matrix_mi(X : np.array, Y : np.array, N_rows : int) -> None:
    # Instantiate the Mutual Information Estimator object
    MI_estimator = MutualInfoEstimator()

    # X_reduced, Y_reduced = self.__pca_dimensionality_reduction(X, Y)
    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    # Initialize the mutual information matrix
    MI_mat = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = X[i]
        Y_row = Y[j]

        MI_estimator.set_inputs(X_row, Y_row)
        # MI_data = self._MI_estimator.kraskov_MI()
        # MI_mat[i, j] = MI_data["MI"]
        MI, _ = MI_estimator.MINE_MI()
        MI_mat[i, j] = MI

    return MI_mat, i_pos, j_pos


def compute_matrix_mi_symmetric(X : np.array, Y : np.array, N_rows : int) -> None:
    # Instantiate the Mutual Information Estimator object
    MI_estimator = MutualInfoEstimator()

    # X_reduced, Y_reduced = self.__pca_dimensionality_reduction(X, Y)
    # Initialize the mutual information matrix
    MI_mat = np.zeros((N_rows, N_rows))
    temp = np.triu_indices(X.shape[0], k=0)
    upper_idx = np.vstack((temp[0], temp[1])).T

    for i, j in upper_idx:
        X_row = X[i]
        Y_row = Y[j]

        MI_estimator.set_inputs(X_row, Y_row)
        # MI_data = MI_estimator.kraskov_MI()
        # MI_mat[i, j] = MI_data["MI"]
        MI, _ = MI_estimator.MINE_MI()
        MI_mat[i, j] = MI
    
    MI_mat = MI_mat + MI_mat.T - np.diag(np.diag(MI_mat))
    return MI_mat
