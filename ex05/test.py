import numpy as np
import importlib
z_score = importlib.import_module('z-score')
from scipy.stats import stats

X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
print(z_score.zscore(X), end='\n\n')

Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
print(z_score.zscore(Y))
