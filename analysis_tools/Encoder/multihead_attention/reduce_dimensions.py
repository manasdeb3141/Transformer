
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple

def reduce_dimensions(X : np.array, Y : np.array) -> np.array:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = scaler.transform(Y)

    # Retain 95% of the variance
    pca_obj = PCA(0.95)
    X_reduced = pca_obj.fit_transform(X_scaled)
    Y_reduced = pca_obj.transform(Y_scaled)
    return X_reduced, Y_reduced
