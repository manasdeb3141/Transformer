
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Classes implemented by this application
from mutual_info_estimator import MutualInfoEstimator


class ArrayMutualInfoEstimator:
    def __init__(self, array_in=None, array_out=None):
        if array_in is None or array_out is None:
            self._array_in = None  
            self._array_out = None  
        else:
            if array_in.shape[0] != array_out.shape[0]:
                raise ValueError("The number of rows in the input and output arrays must be equal.")

            if array_in.shape[1] != array_out.shape[1]:
                raise ValueError("The number of columns in the input and output arrays must be equal.")

            self._array_in = array_in  
            self._array_out = array_out  
    
    def set_data(self, array_in, array_out):
        if array_in.shape[0] != array_out.shape[0]:
            raise ValueError("The number of rows in the input and output arrays must be equal.")

        if array_in.shape[1] != array_out.shape[1]:
            raise ValueError("The number of columns in the input and output arrays must be equal.")

        self._array_in = array_in  
        self._array_out = array_out  

    def estimate_column_mutual_info(self, N_rows = None, N_cols = None):
        # Validate the input arguments
        N_rows, N_cols = self.__validate_inputs(N_rows, N_cols)

        MI_kde = np.zeros(N_cols)
        MI_kraskov = np.zeros(N_cols)
        MI_mine = np.zeros(N_cols)

        for i in range(N_cols):
            kde_mi, kraskov_mi, mine_mi  = self.__estimate_mutual_info(self._array_in[:N_rows, i], self._array_out[:N_rows, i])
            MI_kde[i] = kde_mi
            MI_kraskov[i] = kraskov_mi
            MI_mine[i] = mine_mi

        return MI_kde, MI_kraskov, MI_mine

    def estimate_row_mutual_info(self, N_rows = None, N_cols = None):
        # Validate the input arguments
        N_rows, N_cols = self.__validate_inputs(N_rows, N_cols)

        MI_kde = np.zeros(N_rows)
        MI_kraskov = np.zeros(N_rows)
        MI_mine = np.zeros(N_rows)

        for i in range(N_rows):
            kde_mi, kraskov_mi, mine_mi  = self.__estimate_mutual_info(self._array_in[i, :N_cols], self._array_out[i, :N_cols])
            MI_kde[i] = kde_mi
            MI_kraskov[i] = kraskov_mi
            MI_mine[i] = mine_mi

        return MI_kde, MI_kraskov, MI_mine 

    def estimate_array_mutual_info(self, N_rows = None, N_cols = None):
        # Validate the input arguments
        N_rows, N_cols = self.__validate_inputs(N_rows, N_cols)

        # Extract the elements that are within the specified range
        # of the rows and columns
        data_in = self._array_in[:N_rows, :N_cols]
        data_out = self._array_out[:N_rows, :N_cols]

        # Reshape the data to have a single column
        X = data_in.reshape(-1, 1)
        Y = data_out.reshape(-1, 1)

        # Estimate the mutual information
        kde_mi, kraskov_mi, mine_mi  = self.__estimate_mutual_info(X, Y)
        return kde_mi, kraskov_mi, mine_mi 

    def __estimate_mutual_info(self, X, Y):
        MI_est = MutualInfoEstimator(X, Y)

        # Kernel Density Estimation of Mutual Information
        KDE_pdf, KDE_MI = MI_est.kernel_MI(100)
        kde_mi = KDE_MI["MI"]

        # Kraskov Mutual Information Estimator
        MI_kraskov = MI_est.kraskov_MI()
        kraskov_mi = MI_kraskov["MI"]

        # MINE Mutual Information Estimator
        mine_mi, log = MI_est.MINE_MI()

        return kde_mi, kraskov_mi, mine_mi

    def __validate_inputs(self, N_rows : int, N_cols : int):
        if N_rows is None:
            N_rows = self._array_in.shape[0]
        elif N_rows > self._array_in.shape[0]:
            print(f"Warning: N_rows={N_rows} is greater than the number of rows in the input array ({self._array_in.shape[0]}).")
            N_rows = self._array_in.shape[0]

        if N_cols is None:
            N_cols = self._array_in.shape[1]
        elif N_cols > self._array_in.shape[1]:
            print(f"Warning: N_cols={N_cols} is greater than the number of columns in the input array ({self._array_in.shape[1]}).")
            N_cols = self._array_in.shape[1]

        return N_rows, N_cols

        