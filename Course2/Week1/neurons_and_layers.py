"""
Showing the Linear and Logistic Dense layers are just Regression and that
we have parity with tensorflow for my implementation of Linear and Logistic Regression.
"""
import tensorflow as tf
from datasets import load_tumor_simple
from utils.regression import LinearRegression, LogisticRegression


def main():
    dataset = load_tumor_simple()

    linear_layer = tf.keras.layers.Dense(units=1, input_dim=1, activation="linear", name='linear')
    logistic_layer = tf.keras.layers.Dense(units=1, input_dim=1, activation="sigmoid", name='logistic')
    p_lin = linear_layer(dataset.X_train)
    p_log = logistic_layer(dataset.X_train)

    linear_regression = LinearRegression(x=dataset.X_train, y=dataset.y_train, w=linear_layer.get_weights()[0],
                                         b=linear_layer.get_weights()[1])

    logistic_regression = LogisticRegression(x=dataset.X_train, y=dataset.y_train, w=logistic_layer.get_weights()[0],
                                             b=logistic_layer.get_weights()[1])

    print(f"Linear regression parity with tensorflow: "
          f"{all(abs(linear_regression.predict().reshape(-1, 1) - p_lin.numpy()) <= 0.01)}")
    print(f"Logistic regression parity with tensorflow: "
          f"{all(abs(logistic_regression.predict() - p_log.numpy().reshape(-1)) <= 0.01)}")


if __name__=='__main__':
    main()
