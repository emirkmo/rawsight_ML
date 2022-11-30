import numpy as np
from datasets import Dataset, load_housing_data
from rawsight import ZScoreNorm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from rawsight.cost_functions import least_squares_cost_function, regularized_least_squares_cost_function
from rawsight.optimizers import batch_gradient_descent, regularized_batch_gradient_descent
from rawsight.models import LinearModel

np.random.seed(1)

housing_data = load_housing_data()

# Ensure my output is in the same format as the sklearn output.
xnorm_mine = ZScoreNorm().normalize(housing_data.X_train)
xnorm_sk = StandardScaler().fit_transform(housing_data.X_train)
print(np.all(xnorm_sk[:5] == xnorm_mine[:5]))  # Yes!

# Ensure we round trip: Yes!
print(np.ptp(housing_data.X_train, axis=0))
housing_data.normalize_features()
print(np.ptp(housing_data.X_train, axis=0))
housing_data.denormalize_features()
housing_data.normalize_features()
print(np.ptp(housing_data.X_train, axis=0))

# Fit with SK learn Stochastic Gradient Descent,
# regularized loss, adaptive learning rate ("invscaling"), and L2 regularization.
# with a stop after 5 n_iter_no_change iterations within 0.001 tolerance and epsilon=0.1.
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(housing_data.X_train, housing_data.y_train)

# Fit with mine. Fixed learning rate, L2 regularization.
model = LinearModel(w=1, b=0, n_features=housing_data.X_train.shape[1])
cost_function = least_squares_cost_function
model = batch_gradient_descent(housing_data.X_train, housing_data.y_train, model, cost_function,
                               learning_rate=0.03, max_iter=20000)

# Fit regularized. Fixed learning rate, L2 regularization.
model2 = LinearModel(w=1, b=0, n_features=housing_data.X_train.shape[1])
cost_function = regularized_least_squares_cost_function
model2 = regularized_batch_gradient_descent(housing_data.X_train, housing_data.y_train, model2, cost_function,
                               learning_rate=0.01, max_iter=20000, regularization_param=0.15)
# Compare the results
print(model)
print(model2)
print(sgdr.coef_, sgdr.intercept_)  # Good match!

# make predictions
y_pred_sgd = sgdr.predict(housing_data.X_train)
y_pred_model = model(housing_data.X_train)

print(f"Prediction on training set:\n{y_pred_sgd[:4]}" )
print(f"Target values \n{housing_data.y_train[:4]}")
print(f"Prediction on training set, mine:\n{y_pred_model[:4]}" )
