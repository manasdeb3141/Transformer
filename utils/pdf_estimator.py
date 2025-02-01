from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import numpy as np
import time

class PdfEstimator:
    def __init__(self) -> None:
        super().__init__()
        self._kde = None

    def fit_data(self, samples : np.ndarray) -> None:
        # Create a KernelDensity estimator
        kde = KernelDensity()

        # Define the parameter grid to search
        param_grid = {
            'bandwidth': np.linspace(0.1, 2, 50),
            'kernel': ['gaussian']
            # 'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        }

        # Create a GridSearchCV object
        grid = GridSearchCV(kde, param_grid, cv=5, n_jobs=-1)

        # Fit the GridSearchCV object to the probability distribution samples
        start_time = time.time()
        grid.fit(samples)
        end_time = time.time()
        print(f"GridSearchCV.fit() ran for {end_time-start_time} seconds")

        # Best bandwidth and Kernel Density Estimator
        self._best_params = grid.best_params_
        self._best_kde = grid.best_estimator_
        print("KDE best params:")
        print(self._best_params) 
        print("KDE best estimator:")
        print(self._best_kde) 

    def run(self, pdf_points : np.ndarray) -> np.ndarray:
        # Use the best estimator to make predictions
        log_density = self._best_kde.score_samples(pdf_points) 
        pdf_estimate = np.exp(log_density)
        return pdf_estimate

