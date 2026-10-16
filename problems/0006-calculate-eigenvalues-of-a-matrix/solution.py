import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    matrix = np.array(matrix, dtype=float)
    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalues = np.real_if_close(eigenvalues)
    return sorted([float(x) for x in eigenvalues], reverse=True)