import numpy as np
from utils import LinearModel, LeastSquaresCostFunction, batch_gradient_descent


def main():
    x_train = np.array([1.0, 2.0])  # features
    y_train = np.array([300.0, 500.0])  # target value

    model = LinearModel(w=100, b=0.5)
    cost_function = LeastSquaresCostFunction()

    initial_cost = cost_function(x_train, y_train, model)
    print(f"Initial cost: {initial_cost}, initial model: {model}")

    model = batch_gradient_descent(x_train, y_train, model, cost_function, learning_rate=0.1, max_iter=500)

    final_cost = cost_function(x_train, y_train, model)
    print(f"Final cost: {final_cost}, final model: {model}")


if __name__ == '__main__':
    main()
