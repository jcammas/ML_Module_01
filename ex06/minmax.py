import numpy as np


def minmax(x):
    if not isinstance(x, np.ndarray) or len(x) == 0 or len(x.shape) != 2\
            or x.shape[1] != 1:
        return None
    # Getting min value inside data
    try:
        data_min = np.min(x, axis=0)
    except (TypeError, np.core._exceptions.UFuncTypeError):
        return None
    # Getting the distance between max and min
    data_range = np.max(x, axis=0) - data_min
    # Getting scale from the range -> value to multiply
    # with to fit data inside a specific range
    scale = 1 / data_range
    # Getting minimal value in regard to the scale
    min_ = -data_min * scale
    # Applying scale to every scalar value
    x = x * scale
    # Adding min to stay in between the range boundaries
    x = x + min_
    return x

