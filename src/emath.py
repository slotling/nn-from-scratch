import numpy as np

def sigmoid(z: np.float64 | np.ndarray):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z: np.float64 | np.ndarray):
    return sigmoid(z) * (1.0 - sigmoid(z))

def softmax(x: np.ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy_cost(x: np.ndarray):
    pass

def derivative_cross_entropy_cost_times_derivative_softmax(prediction: np.ndarray, real: np.ndarray):
    return prediction - real