import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')

ALLOWED_TYPES = [int, float, np.int64, np.float64]


class MyLinearRegression:
    """
    Description:
        My personnal lienar regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=100000):
        if not isinstance(alpha, float) or not isinstance(max_iter, int):
            return None
        if alpha <= 0 or max_iter <= 0:
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        if not MyLinearRegression.verif_params(thetas) or type(thetas) is list and len(thetas) != 2 or\
                isinstance(thetas, np.ndarray) and thetas.shape != (2, 1):
            return None
        self.thetas = np.array(thetas).reshape(-1, 1)

    @staticmethod
    def add_intercept(x):
        """
        Adds a column of 1's to the non-empty numpy.array x
        """
        try:
            new_array = np.array(x, dtype=float)
        except (np.core._exceptions.UFuncTypeError, ValueError):
            return None
        intercept_ = np.ones((1, len(x)), dtype=float)
        return np.insert(new_array, 0, intercept_, axis=1)

    def gradient(self, x, y):
        if not MyLinearRegression.verif_params(x, y):
            return None
        if len(x) == 0 or len(y) == 0:
            return None
        x = MyLinearRegression.add_intercept(x)
        y_hat = self.predict_(x)
        try:
            gradients = (1 / len(x)) * np.dot(x.T, np.dot(x, self.thetas) - y)
        except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
            return None
        return gradients


    @staticmethod
    def verif_params(*args):
        for arg in args:
            if not isinstance(arg, list) and not isinstance(arg, np.ndarray):
                return False
            for val in arg:
                if type(val) not in ALLOWED_TYPES:
                    if isinstance(val, np.ndarray):
                        try:
                            tmp = val.sort()
                        except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
                            return False
                        continue
                    return False
        return True

    def fit_(self, x, y):
        if not MyLinearRegression.verif_params(x, y):
            return None
        x = np.array(x)
        y = np.array(y)
        if x.shape[1] != 1 or y.shape[1] != 1:
            return None
        for i in range(self.max_iter):
            # Compute gradients
            gradients = self.gradient(x, y)
            # Update theta in regard to new gradients and learning_rate
            self.thetas = self.thetas - (self.alpha * gradients)
            # Loop this process

    def predict_(self, x):
        if not MyLinearRegression.verif_params(x):
            return None
        y_hat = self.thetas[0] + self.thetas[1] * x
        return y_hat


    @staticmethod
    def mse_(y, y_hat):
        if not MyLinearRegression.verif_params(y, y_hat):
            return None
        if len(y) == 0 or len(y_hat) == 0:
            return None
        y = MyLinearRegression.add_intercept(y)
        y_hat = MyLinearRegression.add_intercept(y_hat)
        loss = np.sum((np.subtract(y, y_hat)**2) * (1 / len(y)))
        return float(loss)

    def plot_hypothesis(self, x, y, y_hat):
        if not MyLinearRegression.verif_params(x, y, y_hat):
            return None
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(x, y, c='#1DEEF4', label='Strue(pills)')
        plt.grid()
        plt.plot(x, y_hat, c='#1DF441', linestyle='dashed')
        plt.scatter(x, y_hat, c='#1DF441', label='Spredict(pills)')
        plt.xlabel('Quantity of blue pill (in micrograms)')
        plt.ylabel('Space driving score')
        plt.legend()
        plt.show()

    def plot_loss_function(self, x, y):
        if not MyLinearRegression.verif_params(x, y):
            return None
        # Settings static theta0 values and linspaced theta1 values
        # in order to generate models with different theta values
        # and perform predictions with different theta values
        # and measure the loss value for each theta value
        theta0_list = [84., 89., 93., 97., 101., 105.]
        n = len(theta0_list)
        colors = plt.cm.Greys(np.linspace(1, 0, n + 1))
        continuous_theta1 = np.arange(-18, -4, 0.01).reshape(-1, 1)
        # Generating len(continuous_theta1) models for each theta0 value
        for index, theta0 in enumerate(theta0_list):
            loss = []
            # Generating a new model for each continuous theta1 value
            for theta1 in continuous_theta1:
                model = MyLinearRegression(np.array([[theta0], [theta1]]))
                loss.append(model.mse_(y, model.predict_(x)))
            plt.plot(continuous_theta1, loss,
                     label="J(${{\Theta_0}}$=$c_{}$, ${{\Theta_1}}$)".format(index),
                     color=colors[index + 1])
        plt.xlabel('${\Theta_1}$')
        plt.ylabel("Cost function J(${\Theta_0}$, ${\Theta_1})$")
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def __str__(self):
        return f'thetas = {str(self.thetas)} || alpha = {self.alpha} || max_iter = {self.max_iter}'

if __name__ == "__main__":
    data = pd.read_csv('are_blue_pills_magics.csv')
    Xpill = np.array(data.Micrograms).reshape(-1, 1)
    Yscore = np.array(data.Score).reshape(-1, 1)
    print("\033[37;1;4mModel with fit and plots\033[0m", end='\n\n')
    model = MyLinearRegression(np.array([[89.], [-8.]]))
    model.fit_(Xpill, Yscore)
    y_hat = model.predict_(Xpill)
    print("Loss :", model.mse_(Yscore, y_hat), end='\n\n')
    model.plot_hypothesis(Xpill, Yscore, y_hat)
    model.plot_loss_function(Xpill, Yscore)

    print('\033[1;4mOther tests :\033[0m', end='\n\n')
    model1 = MyLinearRegression(np.array([[89.], [-8.]]))
    Y_model1 = model1.predict_(Xpill)
    print(model1.mse_(Yscore, Y_model1))
    print(mean_squared_error(Yscore, Y_model1))

    model2 = MyLinearRegression(np.array([[89.], [-6]]))
    Y_model2 = model2.predict_(Xpill)
    print(model2.mse_(Yscore, Y_model2))
    print(mean_squared_error(Yscore, Y_model2))
