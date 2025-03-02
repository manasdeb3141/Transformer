from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
import numpy as np
import time

class PdfEstimator:
    def __init__(self, type : str) -> None:
        super().__init__()
        if type == "statsmodel" or type == "sklearn":
            self._kde_type = type
        else:
            raise ValueError("Invalid type for PdfEstimator")

    def __make_row_vectors(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
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

    def __statsmodel_fit_data(self, samples : np.ndarray) -> None:
        settings = sm.nonparametric.EstimatorSettings(n_jobs=-1)
        self._kde = sm.nonparametric.KDEMultivariate(data=samples, var_type='cc', bw='cv_ml', defaults=settings)
        # print(f"KDE bandwidth: {self._kde.bw}")

    def __statsmodel_fit_data_cond(self, X : np.ndarray, Y : np.ndarray) -> None:
        settings = sm.nonparametric.EstimatorSettings(n_jobs=-1)
        self._kde = sm.nonparametric.KDEMultivariateConditional(endog=Y, 
                                                                exog=X,
                                                                dep_type='c', 
                                                                indep_type='c', 
                                                                bw='cv_ml', defaults=settings)
        print(f"KDE bandwidth: {self._kde.bw}")

    def __statsmodel_pdf(self, pdf_points : np.ndarray) -> np.ndarray:
        return self._kde.pdf(pdf_points)

    def __statsmodel_pdf_cond(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        return self._kde.pdf(endog=Y, exog=X)

    def __sklearn_fit_data(self, samples : np.ndarray) -> None:
        # Create a KernelDensity estimator
        kde = KernelDensity()
        N_tries = 4

        # Initial settings for bandwidth limits
        min_bw = 0.005
        max_bw = 1

        for try_idx in range(N_tries):
            # Define the parameter grid to search
            param_grid = {
                'bandwidth': np.linspace(min_bw, max_bw, 200),
                'kernel': ['gaussian']
                # 'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
            }

            # Create a GridSearchCV object
            grid = GridSearchCV(kde, param_grid, cv=5, n_jobs=-1)

            # Fit the GridSearchCV object to the probability distribution samples
            grid.fit(samples)

            # Best bandwidth and Kernel Density Estimator
            best_params = grid.best_params_
            self._kde = grid.best_estimator_

            if (best_params['bandwidth'] == min_bw):
                min_bw, max_bw = min_bw / (10 ** (try_idx+1)), min_bw+1e-6
            elif (best_params['bandwidth'] == max_bw):
                min_bw, max_bw = max_bw-1e-6, max_bw * (2 ** (try_idx+1))
            else:
                return

        if (best_params['bandwidth'] == min_bw) or (best_params['bandwidth'] == max_bw):
            print("KDE best params:")
            print(best_params) 
            print(f"ERROR: bandwidth = {best_params['bandwidth']} is at the limit after {try_idx} tries")
            raise ValueError("KDE bandwidth is at the limit")


    def __sklearn_pdf(self, pdf_points : np.ndarray) -> np.ndarray:
        # Use the best estimator to make predictions
        log_density = self._kde.score_samples(pdf_points) 
        pdf_estimate = np.exp(log_density)
        return pdf_estimate

    def fit_data(self, samples : np.ndarray) -> None:
        # start_time = time.time()

        if self._kde_type == "statsmodel":
            self.__statsmodel_fit_data(samples)
        else:
            self.__sklearn_fit_data(samples)

        # end_time = time.time()
        # print(f"PdfEstimator fit_data() took {end_time-start_time} seconds to run")

    def pdf(self, pdf_points : np.ndarray) -> np.ndarray:
        # Use the selected KDE estimator to make predictions
        if self._kde_type == "statsmodel":
            pdf_estimate = self.__statsmodel_pdf(pdf_points)
        else:
            pdf_estimate = self.__sklearn_pdf(pdf_points)

        return pdf_estimate

    def fit_data_cond(self, X : np.ndarray, Y: np.ndarray) -> None:
        start_time = time.time()

        if self._kde_type == "statsmodel":
            X_vec, Y_vec = self.__make_row_vectors(X, Y)
            self.__statsmodel_fit_data_cond(X_vec, Y_vec)
        else:
            raise ValueError("Conditional KDE not supported for sklearn")

        end_time = time.time()
        print(f"PdfEstimator fit_data_cond() took {end_time-start_time} seconds to run")

    def pdf_cond(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        if self._kde_type == "statsmodel":
            X_vec, Y_vec = self.__make_row_vectors(X, Y)
            pdf_estimate = self.__statsmodel_pdf_cond(X_vec, Y_vec)
        else:
            raise ValueError("Conditional KDE not supported for sklearn")

        return pdf_estimate
