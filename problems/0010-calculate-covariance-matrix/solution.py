import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    x = np.array(vectors, dtype=float)
    return np.cov(x).tolist()