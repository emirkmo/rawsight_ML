from rawsight.models import LinearModel, add_poly_features
from rawsight.optimizers import batch_gradient_descent
from rawsight import least_squares_cost_function
from rawsight.normalization import ZScoreNorm
import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.arange(0, 20, 1)
    y = np.cos(x / 2)

    x_train = add_poly_features(x, degrees=13)
    norm = ZScoreNorm()
    x_train_norm = norm(x_train)

    model = LinearModel(w=1, b=0, n_features=x_train.shape[-1])
    cost_function = least_squares_cost_function

    model = batch_gradient_descent(x_train_norm, y, model, cost_function, learning_rate=0.1, max_iter=100000)
    print(model)
    for i in range(x_train.shape[0]):
        print(f"pred: {model(np.atleast_1d(x_train_norm[i]))}, target: {y[i]}")

    plt.plot(x, y, "s")
    plt.plot(x, model(x_train_norm), "o-")
    plt.show(block=True)


if __name__ == "__main__":
    main()
