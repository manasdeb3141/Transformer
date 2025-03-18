import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from typing import List, Tuple

def compute_bhattacharya_coefficient(P_X_mat: np.ndarray, P_Y_mat: np.ndarray) -> float:
    N_rows = P_X_mat.shape[0]
    N_cols = P_X_mat.shape[1]

    # Initialize the bhattacharya distance matrix
    bhattacharya_coeff = np.zeros((N_rows, N_rows))

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    for i, j in tqdm(ij_pos):
        P_X = P_X_mat[i]
        P_Y = P_Y_mat[j]
        BC = np.sum(np.sqrt(P_X * P_Y))
        bhattacharya_coeff[i, j] = BC

    return bhattacharya_coeff


def compute_wasserstein_distance(P_X_mat: np.ndarray, P_Y_mat: np.ndarray) -> float:
    N_rows = P_X_mat.shape[0]
    N_cols = P_X_mat.shape[1]

    # Initialize the wasserstein distance matrix
    wassersten_dist = np.zeros((N_rows, N_rows))

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    for i, j in tqdm(ij_pos):
        P_X = P_X_mat[i]
        P_Y = P_Y_mat[j]
        dist = wasserstein_distance(np.arange(N_cols), np.arange(N_cols), P_X, P_Y)
        wassersten_dist[i, j] = dist

    return wassersten_dist
