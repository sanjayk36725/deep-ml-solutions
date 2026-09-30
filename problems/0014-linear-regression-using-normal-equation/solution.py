import numpy as np

def linear_regression_normal_equation(X, y):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    return [round(float(v), 4) for v in theta]