import numpy as np

def sigmoid(z: np.float64 | np.ndarray):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z: np.float64 | np.ndarray):
    return sigmoid(z) * (1.0 - sigmoid(z))

def softmax(z: np.ndarray):
    e_x = np.exp(z - np.max(z))
    return e_x / e_x.sum(axis=0)