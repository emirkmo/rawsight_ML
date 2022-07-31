import numpy as np
import pandas as pd
from utils import least_squares_cost_function
from datasets import Dataset
from utils.models import LinearModel
from utils.optimizers import batch_gradient_descent


def run(housing_dataset: Dataset, learning_rate: float = 0.1):
    # Run multi-feature linear regression with batch gradient descent
    model = LinearModel(w=0, b=0, n_features=housing_dataset.num_features)
    cost_function = least_squares_cost_function
    initial_cost = cost_function(housing_dataset.X_train, housing_dataset.y_train, model)
    print(f"Initial cost: {initial_cost}, initial model: {model}")

    model = batch_gradient_descent(housing_dataset.X_train, housing_dataset.y_train, model, cost_function,
                                   learning_rate=learning_rate, max_iter=1000)

    final_cost = cost_function(housing_dataset.X_train, housing_dataset.y_train, model)
    print(f"Final cost: {final_cost}, final model: {model}")
    return model


def main():
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    df = pd.DataFrame(X_train, columns=["size", "bedrooms", "floors", "age"])
    df["price"] = np.array([460, 232, 178])
    housing_dataset = Dataset(df=df, name="house prices", target_name="price")
    housing_dataset.split_train_test(0.0)

    model = run(housing_dataset, learning_rate=5e-7)
    print(f"pred: {model(np.atleast_1d(housing_dataset.X_train))}, target: {housing_dataset.y_train}")

    print("Expected Result:")
    print("w, b found by gradient descent: [ 0.2 0. -0.01 -0.07], -0.00")
    print("prediction: 426.19, target value: 460")
    print("prediction: 286.17, target value: 232")
    print("prediction: 171.47, target value: 178")


if __name__ == "__main__":
    main()
