import numpy as np

def swish(x):
    return x / (1 + np.exp(-x))