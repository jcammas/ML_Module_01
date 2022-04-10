import numpy as np
import warnings
warnings.filterwarnings('ignore')

def simple_gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    if x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1):
        return None
    gradients = np.zeros((2, 1))
    try:
        y_hat = np.sum((theta[0], theta[1] * x))
    except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
        return None
    for i in range(len(x)):
        try:
            gradients[0] += y_hat[i] - y[i]
            gradients[1] += (y_hat[i] - y[i]) * x[i]
        except (np.core._exceptions.UFuncTypeError, TypeError):
            return None
    gradients = gradients * (1 / len(x))
    return gradients
