import numpy as np

def sigmoid(z: np.float64 | np.ndarray):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z: np.float64 | np.ndarray):
    return sigmoid(z) * (1.0 - sigmoid(z))