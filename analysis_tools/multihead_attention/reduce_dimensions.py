
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple

def reduce_dimensions(self, X : np.array, Y : np.array) -> Tuple[np.array, np.array]:
    # Retain 95% of the variance
    pca_X = PCA(0.95)
    pca_X.fit(X)
    X_n_components = pca_X.n_components_

    pca_Y = PCA(0.95)
    pca_Y.fit(Y)
    Y_n_components = pca_Y.n_components_

    # Get the maximum number of components to retain
    N_components = max(X_n_components, Y_n_components)

    # Fit the PCA model to the data and transform the data.
    # Do the inverse transform to get the original data back with reduced dimensions
    pca_X = PCA(n_components=N_components)
    # X_reduced = pca_X.fit_transform(X)
    transf_data = pca_X.fit_transform(X)
    X_reduced = pca_X.inverse_transform(transf_data)

    pca_Y = PCA(n_components=N_components)
    # Y_reduced = pca_Y.fit_transform(Y)
    transf_data = pca_Y.fit_transform(Y)
    Y_reduced = pca_Y.inverse_transform(transf_data)

    return X_reduced, Y_reduced
