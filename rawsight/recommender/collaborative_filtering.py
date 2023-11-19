import numpy as np
import numpy.typing as npt


def cofi_cost_func(
    X: npt.NDArray[np.number],
    W: npt.NDArray[np.number],
    b: npt.NDArray[np.number],
    Y: npt.NDArray[np.number],
    R: npt.NDArray[np.number],
    lam: float,
) -> float:
    """Return cost with regularization using numpy for collaborative learning
    Args:
      X np(num_feature_samples, num_features)): matrix of feature samples
      W np(num_parameter_samples, num_features)) : matrix of parameter samples
      b np(1, num_parameter_samples)            : constant parameter vector per param sample.
      Y np(num_feature_samples,num_parameter_samples) : matrix of pars per feature sample
      R np(num_feature_samples,num_parameter_samples) : R(i, j) = 1 if feature sample has parameters.
      lam (float): regularization parameter

    Simples example is X features of movies and W is features of user ratings (for movies)
    Y is matrix of user ratings for each movie and R just records if a user rated a movie.
    """
    # Regularization is simple and applies to all values
    regularization: float = (np.sum(W**2) + np.sum(X**2)) * (lam / 2)

    # Linear regression analog vectorized implementation.
    cost: float = np.sum((R * (np.dot(X, W.T) + b - Y)) ** 2) / 2

    return cost + regularization
