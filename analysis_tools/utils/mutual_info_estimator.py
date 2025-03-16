import numpy as np
import matplotlib.pyplot as plt

from pdf_estimator import PdfEstimator
from scipy.integrate import simpson
from scipy.stats import wasserstein_distance
from typing import Tuple

from npeet import entropy_estimators as ee
from MINE_estimator import MINE_estimator

class MutualInfoEstimator:
    def __init__(self, X : np.ndarray = None, Y : np.ndarray = None) -> None:
        super().__init__()

        # Default KDE estmator module
        self._KDE_est_module = 'statsmodel'

        # X and Y are expected to be row vectors (1-D arrays)
        if (X is not None) and (Y is not None):
            if len(X.shape) < 2:
                self._X = np.expand_dims(X, axis=1)
                self._Y = np.expand_dims(Y, axis=1)
            elif len(X.shape) == 2:
                if X.shape[1] == 1 and Y.shape[1] == 1:
                    self._X = X
                    self._Y = Y
                else:
                    raise ValueError("X and Y should be row vectors")
        else:
            self._X = None
            self._Y = None


    def __estimate_pdf(self, num_points : int, same_range=False, continuous : bool = True) -> np.ndarray:
        xy_samples = np.hstack([self._X, self._Y])

        min_X = np.amin(self._X)
        max_X = np.amax(self._X)
        min_Y = np.amin(self._Y)
        max_Y = np.amax(self._Y)

        pdf_estimator = PdfEstimator(self._KDE_est_module)
        pdf_estimator.fit_data(xy_samples)

        if same_range:
            min_X = min_Y = min(min_X, min_Y)
            max_X = max_Y = max(max_X, max_Y)

        x = np.linspace(min_X, max_X, num_points)
        y = np.linspace(min_Y, max_Y, num_points)

        xpos, ypos = np.meshgrid(x, y)
        xy_samples = np.vstack([xpos.ravel(), ypos.ravel()]).T
        pdf_est = pdf_estimator.pdf(xy_samples)

        # Joint and Marginal probabilities
        P_XY = pdf_est.reshape(num_points, num_points)/np.sum(pdf_est)

        if continuous:
            # Integration of continuous points using Simpson's rule
            P_X = simpson(P_XY, axis=0)
            P_Y = simpson(P_XY, axis=1)
        else:
            # Summation of discrete points
            P_X = np.sum(P_XY, axis=0)
            P_Y = np.sum(P_XY, axis=1)

        xpos = xy_samples[:,0]
        ypos = xy_samples[:,1]
        x_grid = xpos.reshape(num_points, num_points)
        y_grid = ypos.reshape(num_points, num_points)

        xpos = x_grid[0,:]
        ypos = y_grid[:,0]

        pdf_data = dict()
        pdf_data["P_XY"] = P_XY
        pdf_data["P_X"]  = P_X
        pdf_data["P_Y"]  = P_Y
        pdf_data["xpos"]  = xpos
        pdf_data["ypos"]  = ypos
        pdf_data["x_grid"] = x_grid
        pdf_data["y_grid"] = y_grid

        return pdf_data

    def __estimate_pmf(self, N_bins : int) -> np.ndarray:
        H, xedges, yedges = np.histogram2d(self._X, self._Y, bins=N_bins, density=False)
        P_XY = H.T/np.sum(H.T)

        P_X = P_XY.sum(axis=0)
        P_Y = P_XY.sum(axis=1)

        # Create a dictionary for returning the group of values
        pmf_data = dict()
        pmf_data["P_XY"] = P_XY
        pmf_data["P_X"]  = P_X
        pmf_data["P_Y"]  = P_Y
        pmf_data["xedges"] = xedges
        pmf_data["yedges"] = yedges

        return pmf_data

    def __calculate_entropy(self, P_XY : np.ndarray, P_X : np.ndarray, P_Y : np.ndarray, continuous : bool) -> np.ndarray:
        if continuous:
            #
            # Numerical integration of continuous points
            #

            # Masked log2 operation to replace output of log2(0) with 0
            H_XY = -1 * simpson(simpson(P_XY * np.ma.log2(P_XY).filled(0), axis=0))
            H_X =  -1 * simpson(P_X * np.ma.log2(P_X).filled(0))
            H_Y =  -1 * simpson(P_Y * np.ma.log2(P_Y).filled(0))
        else:
            #
            # Sumation of discrete points
            #

            # Masked log2 operation to replace output of log2(0) with 0
            H_XY = -np.sum(P_XY * np.ma.log2(P_XY).filled(0))
            H_X = -np.sum(P_X * np.ma.log2(P_X).filled(0))
            H_Y = -np.sum(P_Y * np.ma.log2(P_Y).filled(0))

        return H_X, H_Y, H_XY


    def __normalized_MI(self, MI, H_X, H_Y):
        norm_MI = MI / max(H_X, H_Y)
        return norm_MI

    def KL_divergence(self, P_X : np.ndarray, P_Y : np.ndarray) -> float:
        # Calculate the Kullback-Leibler divergence
        P_Y[P_Y == 0] = np.finfo(float).eps
        # KL_div = simpson(np.squeeze(P_X * np.ma.log2(P_X / P_Y).filled(0)))
        KL_div = np.sum(np.squeeze(P_X * np.ma.log2(P_X / P_Y).filled(0)))

        if KL_div < 0:
            KL_div = 0

        return KL_div    

    def JS_divergence(self, P_X : np.ndarray, P_Y : np.ndarray) -> float:
        JS_div = (0.5 * self.KL_divergence(P_X, (P_X + P_Y)/2)) + (0.5 * self.KL_divergence(P_Y, (P_X + P_Y)/2))
        return JS_div

    def pdf_softmax(self, beta=1.0) -> Tuple[np.ndarray, np.ndarray]:
        P_X = np.exp(beta * self._X)/np.sum(np.exp(beta * self._X))
        P_Y = np.exp(beta * self._Y)/np.sum(np.exp(beta * self._Y))

        return P_X, P_Y

    def wasser_dist(self) -> float:
        wasserstein_distance(np.arange(self._X.shape[0]), np.arange(self._Y.shape[0]), self._X, self._Y)

    def set_inputs(self, X : np.ndarray, Y : np.ndarray) -> None:
        if len(X.shape) < 2:
            self._X = np.expand_dims(X, axis=1)
            self._Y = np.expand_dims(Y, axis=1)
        elif len(X.shape) == 2:
            if X.shape[1] == 1 and Y.shape[1] == 1:
                self._X = X
                self._Y = Y
            else:
                raise ValueError("X and Y should be row vectors")

    def kernel_MI(self, N_points : int = 100, KDE_module : str = 'sklearn', continuous=True) -> float:
        self._KDE_est_module = KDE_module

        if continuous:
            # Calculate the PDF based on Kernel Density Estimation
            prob_data = self.__estimate_pdf(N_points, same_range=False, continuous=True)
            P_XY = prob_data["P_XY"]
            P_X = prob_data["P_X"]
            P_Y = prob_data["P_Y"]
        else:
            # Calculate the PMF based on histogram bins
            prob_data = self.__estimate_pmf(N_points)
            P_XY = prob_data["P_XY"]
            P_X = prob_data["P_X"]
            P_Y = prob_data["P_Y"]
            
        # Calculate the Joint Entropy and the marginal entropy of the random vectors
        H_X, H_Y, H_XY = self.__calculate_entropy(P_XY, P_X, P_Y, continuous=continuous)
            
        # Mutual Information
        MI = H_X + H_Y - H_XY

        # Lower bound the mutual information and entropies to zero if it is negative
        if MI < 0:
            MI = 0

        if H_X < 0:
            H_X = 0

        if H_Y < 0:
            H_Y = 0

        if H_XY < 0:
            H_XY = 0

        # Create a dictionary for returning the group of values
        MI_data = dict()
        MI_data["H_X"] = H_X
        MI_data["H_Y"] = H_Y
        MI_data["H_XY"] = H_XY
        MI_data["MI"] = MI

        return prob_data, MI_data

    def kraskov_entropy(self, X=None) -> float:
        if X is None:
            X = self._X

        default_base=2
        H = ee.entropy(X, k=3, base=default_base)
        if H < 0:
            H = 0

        return H


    def kraskov_MI(self) -> float:
        default_base=2
        if len(self._X.shape) > 32:
            k=8
        else:
            k=3

        H_X = ee.entropy(self._X, k=k, base=default_base)
        H_Y = ee.entropy(self._Y, k=k, base=default_base)
        H_XY = ee.entropy(np.hstack([self._X, self._Y]), k=k, base=default_base)
        # MI = ee.mi(self._X, self._Y, k=k, base=default_base, alpha=0.25)
        MI = ee.mi(self._X, self._Y, k=k, base=default_base)
        MI_entropy = H_X + H_Y - H_XY

        # Saturate the mutual information to zero if it is negative
        if MI < 0:
            MI = 0

        if H_X < 0:
            H_X = 0

        if H_Y < 0:
            H_Y = 0

        if H_XY < 0:
            H_XY = 0

        # Create a dictionary for returning the group of values
        MI_data = dict()
        MI_data["H_X"] = H_X
        MI_data["H_Y"] = H_Y
        MI_data["H_XY"] = H_XY
        MI_data["MI_from_entropy"] = MI_entropy
        MI_data["MI"] = MI

        return MI_data

    
    def MINE_MI(self) -> float:
        MINE_est = MINE_estimator(self._X, self._Y)
        MI, run_log = MINE_est.run()

        # Saturate the mutual information to zero if it is negative
        if MI < 0:
            MI = 0

        return (MI * np.log2(np.exp(1))), run_log


    def BC_stats(self, softmax_pdf = True, N_points=100, continuous=True) -> dict:
        # Compute the Battacharyya Coefficient
        if softmax_pdf:
            P_X, P_Y = self.pdf_softmax()
        else:
            prob_data = self.__estimate_pdf(N_points, same_range=True, continuous=True)
            P_X = prob_data["P_X"]
            P_Y = prob_data["P_Y"]

        # Compute the Battacharyya Coefficient
        BC = simpson(np.sqrt(P_X * P_Y))

        # Compute the Bhattacharyya distance
        if BC == 0:
            BD = np.inf
        else:
            BD = -np.log(BC)

        BC_data = dict()
        BC_data["BC"] = BC
        BC_data["BD"] = BD

        return BC_data