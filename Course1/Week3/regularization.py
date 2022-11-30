from datasets import load_housing_data
from rawsight import LinearModel
from rawsight.cost_functions import regularized_logistic_cost_function, regularized_least_squares_cost_function
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDRegressor
from logistic_regression import run_logistic_regression
from rawsight.optimizers import regularized_batch_gradient_descent
np.random.seed(1)
X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0., 0., 0., 1., 1., 1.])

lr_model = LogisticRegression()
lr_model.fit(X, y)
y_pred = lr_model.predict(X)
print("Prediction on training set:", y_pred)
print("Accuracy on training set:", lr_model.score(X, y))
print("pars:", lr_model.coef_, lr_model.intercept_)
print("-" * 100)

my_model = run_logistic_regression(X, y, learning_rate=0.3, max_iter=20000, w=(1, 1), b=-1,
                                   cost_function=regularized_logistic_cost_function)
y_pred = my_model(X)
print("Prediction on training set:", y_pred)
print("pars:", my_model)


# SKlearn Linear Regression
housing_data = load_housing_data()

# Ensure my output is in the same format as the sklearn output.
housing_data.normalize_features()
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(housing_data.X_train, housing_data.y_train)

# Fit with mine. Fixed learning rate, L2 regularization.
model = LinearModel(w=1, b=-2, n_features=housing_data.X_train.shape[1])
cost_function = regularized_least_squares_cost_function
model = regularized_batch_gradient_descent(housing_data.X_train, housing_data.y_train, model, cost_function,
                                           learning_rate=0.03, max_iter=20000, regularization_param=0.15)
# Compare the results
print(model)
print(sgdr.coef_, sgdr.intercept_)  # Good match!

# make predictions
y_pred_sgd = sgdr.predict(housing_data.X_train)
y_pred_model = model(housing_data.X_train)

print(f"Prediction on training set:\n{y_pred_sgd[:4]}" )
print(f"Target values \n{housing_data.y_train[:4]}")
print(f"Prediction on training set, mine:\n{y_pred_model[:4]}" )
