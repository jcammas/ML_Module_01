import numpy as np
from scipy.stats import stats


def zscore(x):
    if not isinstance(x, np.ndarray):
        return None
    if len(x) == 0 or len(x.shape) != 2 or x.shape[1] != 1:
        return None
    try:
        mean = np.mean(x)
    except (np.core._exceptions.UFuncTypeError, TypeError):
        return None
    std = np.std(x)
    zscores = np.array([(value - mean) / std for value in x])
    return zscores

