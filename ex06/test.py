import numpy as np
from minmax import minmax

X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
X = np.array([[0, 15, -9, 7, 12, 3,-21]])
#X = np.array([[]])
#X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],['a']])
#X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[]])
print(minmax(X))

Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
print(minmax(Y))
