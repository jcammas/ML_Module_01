import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

print('\033[37;1;4mFirst linear regression model\033[0m', end='\n\n')
lr1 = MyLR([2, 0.7])
print('lr1 model :\n', lr1, end='\n\n')
print("Prediction :", lr1.predict_(x), end='\n\n')

print("Loss elem :\n", lr1.loss_elem_(y, lr1.predict_(x)), end='\n\n')

print("Loss :", lr1.loss_(y, lr1.predict_(x)), end='\n\n')

print('\033[37;1;4mSecond linear regression model\033[0m', end='\n\n')
lr2 = MyLR([1, 1], 5e-8, 1500000)
print('lr2 model :\n', lr2, end='\n\n')
lr2.fit_(x, y)
print("Thetas after fit :\n", lr2.thetas, end='\n\n')
print("Prediction after fit :\n",lr2.predict_(x), end='\n\n')

print("Loss elem using lr2 :\n", MyLR.loss_elem_(y, lr2.predict_(x)), end='\n\n')
print("Loss using lr2 :\n", MyLR.loss_(y, lr2.predict_(x)), end='\n\n')
