import numpy as np
from vec_gradient import *

x = np.array([[12.4956442],[ 21.5007972],[ 31.5527382],[ 48.9145838],[ 57.5088733]])
y = np.array([[37.4013816],[ 36.1473236],[ 45.7655287],[ 46.6793434],[ 59.5585554]])
#y = np.array([[37.4013816],[ 36.1473236],[ 45.7655287],[ 46.6793434],['a']])
#y = np.array([[37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]])

theta1 = np.array([[2],[ 0.7]])
#theta1 = np.array([])
#theta1 = np.array([[]])
#theta1 = np.array([[2, 0.7]])
#theta1 = np.array([[2],['a']])
#theta1 = np.array([[2]])
print(gradient(x, y, theta1))

theta2 = np.array([[1], [-0.4]])
print(gradient(x, y, theta2))
