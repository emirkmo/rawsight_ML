from utils.linear_regression import LinearModel
from utils.cost_functions import LeastSquaresCostFunction
import numpy as np


def main():
    x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
    y_train = np.array([250, 300, 480, 430, 630, 730, ])
    cost = LeastSquaresCostFunction()
    model = LinearModel(w=100, b=2.)
    print(cost(x_train, y_train, model))
    model = LinearModel(w=209, b=2.4)
    print(cost(x_train, y_train, model))
    model = LinearModel(w=209, b=2.0)
    print(cost(x_train, y_train, model))


if __name__ == '__main__':
    main()
