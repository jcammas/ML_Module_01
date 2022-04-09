import numpy as np
from vec_gradient import add_intercept, gradient

def fit_(x, y, theta, alpha, max_iter):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray) or\
            not isinstance(alpha, float) or not isinstance(max_iter, int):
                return None
    if x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1):
        return None
    if max_iter <= 0:
        return None
    for i in range(max_iter):
        # Compute gradients
        gradients = gradient(x, y, theta)
        # Update theta in regard to new gradients and learning_rate
        theta = theta - (alpha * gradients)
        # Loop this process
    return theta

def predict(x, theta):
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if theta.shape != (2, 1) or x.shape[1] != 1 or len(x) == 0:
        return None
    return np.sum((theta[0], theta[1] * x))
