import numpy as np
import warnings
warnings.filterwarnings('ignore')

ALLOWED_TYPES = [int, float, np.int64, np.float64]


class MyLinearRegression:
    """
    Description:
        My personnal lienar regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
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
        except ValueError:
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
        gradients = (1 / len(x)) * np.dot(x.T, np.dot(x, self.thetas) - y)
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
        x = np.array(x)
        if x.shape[1] != 1 or len(x) == 0:
            return None
        x = MyLinearRegression.add_intercept(np.array(x))
        try:
            y_hat = x @ self.thetas
        except (np.core._exceptions.UFuncTypeError, ValueError):
            return None
        return y_hat

    @staticmethod
    def loss_elem_(y, y_hat):
        if not MyLinearRegression.verif_params(y, y_hat):
            return None
        if len(y) == 0 or len(y_hat) == 0:
            return None
        loss = np.zeros((len(y), 1))
        for i in range(len(y)):
            try:
                loss[i] = (y[i] - y_hat[i])**2
            except np.core._exceptions.UFuncTypeError:
                return None
        return loss

    @staticmethod
    def loss_(y, y_hat):
        if not MyLinearRegression.verif_params(y, y_hat):
            return None
        if len(y) == 0:
            return None
        loss = 0.
        for i in range(len(y)):
            try:
                loss += (y_hat[i] - y[i])**2
            except (TypeError, np.core._exceptions.UFuncTypeError):
                return None
        loss /= 2 * len(y)
        return float(loss)

    def __str__(self):
        return f'thetas = {str(self.thetas)} || alpha = {self.alpha} || max_iter = {self.max_iter}'
