import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    if len(a) == 0 or len(a[0]) != len(b):
        return -1

    return np.dot(np.array(a), np.array(b)).tolist()