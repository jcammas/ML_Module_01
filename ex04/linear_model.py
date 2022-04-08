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
        if type(x) not in ALLOWED_TYPES:
            return None
        y_hat = self.thetas[0] + self.thetas[1] * x
        return y_hat


    @staticmethod
    def loss_(y, y_hat):
        if not MyLinearRegression.verif_params(y, y_hat):
            return None
        if len(y) == 0 or len(y_hat) == 0:
            return None
        self.fit(
        y = MyLinearRegression.add_intercept(y)
        y_hat = MyLinearRegression.add_intercept(y_hat)
        loss = np.su
        return float(loss)

    def __str__(self):
        return f'thetas = {str(self.thetas)} || alpha = {self.alpha} || max_iter = {self.max_iter}'

if __name__ == "__main__":
    data = pd.read_csv('are_blue_pills_magics.csv')
    Xpill = np.array(data.Micrograms).reshape(-1, 1)
    Yscore = np.array(data.Score).reshape(-1, 1)

    model1 = MyLinearRegression(np.array([[89.], [-8.]]))
    Y_model1 = model1.predict_(Xpill)
    print(model1.mse_(Yscore, Y_model1))
    print(mean_squarred_error(Yscore, Y_model1))

    model2 = MyLinearRegression(np.array([[89.], [-6]]))
    Y_model2 = model2.predict_(Xpill)
    print(model2.mse_(Yscore, Y_model2))
    print(mean_squarred_error(Yscore, Y_model2))
