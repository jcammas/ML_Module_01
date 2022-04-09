import numpy as np
from scipy.stats import stats


def zscore(x):
    if not isinstance(x, np.ndarray):
        return None
    if len(x) == 0 or x.shape[1] != 1:
        return None
    mean = np.mean(x)
    std = np.std(x)
    zscores = np.array([(value - mean) / std for value in x])
    return zscores


if __name__ == "__main__":
    X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
    print(zscore(X))
    Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
    print(zscore(Y))
