
import sys
sys.path.append('../../..')
sys.path.append('../../utils')

import numpy as np
from tqdm import tqdm

from mutual_info_estimator import MutualInfoEstimator


def make_col_vectors(X : np.ndarray, Y : np.ndarray) -> np.ndarray:
    if X.shape != Y.shape:
        raise ValueError("X and Y should have the same shape")

    # X and Y are expected to be (1-D numpy arrays)
    if len(X.shape) < 2:
        X_val = np.expand_dims(X, axis=1)
        Y_val = np.expand_dims(Y, axis=1)
    elif len(X.shape) == 2:
        if X.shape[1] == 1 and Y.shape[1] == 1:
            X_val = X
            Y_val = Y
        elif X.shape[0] == 1 and Y.shape[0] == 1:
            X_val = X.T
            Y_val = Y.T
        else:
            raise ValueError("X and Y should be 1-D vectors")
    else:
        raise ValueError("X and Y should be 1-D vectors") 

    return X_val, Y_val

def compute_mutual_info(X_mat: np.ndarray, Y_mat: np.ndarray, symmetric=False) -> np.ndarray:
    N_rows = X_mat.shape[0]

    # Mutual information estimator type
    MI_est_type = "kraskov"
    # MI_est_type = "kernel"

    # Initialize the mutual information matrix
    MI_estimate = np.zeros((N_rows, N_rows))

    if symmetric:
        # Create an array with the upper trangular matrix indices
        # Since MI is symmetric, we only need to compute the upper triangular matrix
        temp = np.triu_indices(N_rows, k=0)
        ij_pos = np.vstack((temp[0], temp[1])).T
    else:
        i = np.arange(0, N_rows, 1)
        j = np.arange(0, N_rows, 1)
        i_pos, j_pos = np.meshgrid(i, j)
        ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    for i, j in tqdm(ij_pos):
        X = X_mat[i]
        Y = Y_mat[j]
        MI_estimator = MutualInfoEstimator(X, Y)

        match MI_est_type:
            case "kraskov":
                MI_data = MI_estimator.kraskov_MI()
                MI = MI_data["MI"]

            case "kernel":
                _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
                MI = MI_data["MI"]

            case "MINE":
                MI, _ = MI_estimator.MINE_MI()

        MI_estimate[i, j] = MI

    if symmetric:
        # The upper and lower triangular matrix values are the same
        MI_estimate = MI_estimate + MI_estimate.T - np.diag(np.diag(MI_estimate))

    return MI_estimate


def KDE_mutual_info(X_mat: np.ndarray, Y_mat: np.ndarray, symmetric=False) -> np.ndarray:
    N_rows = X_mat.shape[0]

    # Initialize the mutual information matrix
    MI_estimate = np.zeros((N_rows, N_rows))

    # The probability matrix is a 2-D array. Each element contains a dictionary 
    # of probability values for the joint and marginal distributions, entropy, and mutual information
    P_matrix = [[dict() for _ in range(N_rows)] for _ in range(N_rows)]

    if symmetric:
        # Create an array with the upper trangular matrix indices
        # Since MI is symmetric, we only need to compute the upper triangular matrix
        temp = np.triu_indices(N_rows, k=0)
        ij_pos = np.vstack((temp[0], temp[1])).T
    else:
        i = np.arange(0, N_rows, 1)
        j = np.arange(0, N_rows, 1)
        i_pos, j_pos = np.meshgrid(i, j)
        ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    for i, j in tqdm(ij_pos):
        X = X_mat[i]
        Y = Y_mat[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        prob_data, MI_data = MI_estimator.kernel_MI(same_range=True, KDE_module='sklearn', continuous=True)
        P_matrix[i][j] = dict(prob_data=prob_data, MI_data=MI_data, i=i, j=j)
        MI = MI_data["MI"]
        MI_estimate[i, j] = MI

    if symmetric:
        # The upper and lower triangular matrix values are the same
        MI_estimate = MI_estimate + MI_estimate.T - np.diag(np.diag(MI_estimate))

        # Copy the P_matrix entries to the lower triangular matrix
        lower_indices = np.tril_indices_from(MI_estimate, k=-1)
        ij_pos = np.vstack((lower_indices[0], lower_indices[1])).T
        for i, j in ij_pos:
            P_matrix[i][j] = P_matrix[j][i]

    return P_matrix, MI_estimate


def KDE_mutual_info_vec(X_vec: np.ndarray, Y_vec: np.ndarray) -> np.ndarray:
    X, Y = make_col_vectors(X_vec, Y_vec)

    N_rows = X_vec.shape[0]

    # Initialize the mutual information matrix
    MI_estimate = np.zeros((N_rows, 1))

    # The probability matrix is a 2-D array. Each element contains a dictionary 
    # of probability values for the joint and marginal distributions, entropy, and mutual information
    P_matrix = [dict() for _ in range(N_rows)]

    for i in tqdm(range(N_rows)):
        MI_estimator = MutualInfoEstimator(X, Y)
        prob_data, MI_data = MI_estimator.kernel_MI(same_range=True, KDE_module='sklearn', continuous=True)
        P_matrix[i] = dict(prob_data=prob_data, MI_data=MI_data)
        MI = MI_data["MI"]
        MI_estimate[i] = MI

    return P_matrix, MI_estimate