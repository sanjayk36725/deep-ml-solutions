import numpy as np

def train(X_train, y_train, X_val, y_val):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    # Add bias column
    X = np.c_[np.ones(X_train.shape[0]), X_train]

    # Initialize weights
    weights = np.zeros(X.shape[1], dtype=float)

    learning_rate = 0.1
    epochs = 2000

    for _ in range(epochs):
        z = X @ weights
        z = np.clip(z, -50, 50)

        probabilities = 1.0 / (1.0 + np.exp(-z))

        gradient = (X.T @ (probabilities - y_train)) / X.shape[0]

        weights -= learning_rate * gradient

    def predict(X):
        X = np.asarray(X, dtype=float)
        X = np.c_[np.ones(X.shape[0]), X]

        z = X @ weights
        z = np.clip(z, -50, 50)

        probabilities = 1.0 / (1.0 + np.exp(-z))

        return (probabilities >= 0.5).astype(int)

    return predict