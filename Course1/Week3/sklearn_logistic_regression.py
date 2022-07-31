from logistic_regression import run_logistic_regression

import numpy as np
from sklearn.linear_model import LogisticRegression


X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0., 0., 0., 1., 1., 1.])

lr_model = LogisticRegression()
lr_model.fit(X, y)
y_pred = lr_model.predict(X)
print("Prediction on training set:", y_pred)
print("Accuracy on training set:", lr_model.score(X, y))
print("pars:", lr_model.coef_, lr_model.intercept_)
print("-" * 100)
my_model = run_logistic_regression(X, y, learning_rate=10, max_iter=10000, w=(1,), b=-2)
y_pred = my_model(X)
print("Prediction on training set:", y_pred)
print("pars:", my_model)
