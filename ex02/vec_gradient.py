import numpy as np
import warnings
warnings.filterwarnings('ignore')


def add_intercept(x):
    """
    Adds a column of 1's to the non-empty numpy.array x
    """
    if not isinstance(x, np.ndarray):
        return None
    if len(x.shape) != 2 or x.shape[1] != 1:
        return None
    try:
        new_array = np.array(x, dtype=float)
    except ValueError:
        return None
    intercept_ = np.ones((1, len(x)), dtype=float)
    return np.insert(new_array, 0, intercept_, axis=1)

def gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    if x.shape[1] != 1 or y.shape[1] != 1 or theta.shape != (2, 1):
        return None
    try:
        # Adding intercept to perform matrix dot products
        x = add_intercept(x)
        # Getting predictions
        y_hat = np.sum((theta[0], theta[1] * y))
        # vectorized gradients computation
        gradients = (1 / len(x)) * np.dot(x.T, np.dot(x, theta) - y)
    except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
        return None
    return gradients
