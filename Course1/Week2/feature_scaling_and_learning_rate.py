from utils import LeastSquaresCostFunction, MaxNorm, MeanNorm, ZScoreNorm
from datasets import load_housing_data, Dataset
from utils.models import LinearModel
from utils.optimizers import batch_gradient_descent


def load(norm: bool = False):
    # Load and normalize the housing data
    housing_dataset = load_housing_data()
    if norm:
        housing_dataset.normalize_features(ZScoreNorm())
    return housing_dataset


def run(housing_dataset: Dataset, learning_rate: float = 0.1):
    # Run multi-feature linear regression with batch gradient descent
    model = LinearModel(w=1, b=0, n_features=housing_dataset.num_features)
    cost_function = LeastSquaresCostFunction()
    initial_cost = cost_function(housing_dataset.X_train, housing_dataset.y_train, model)
    print(f"Initial cost: {initial_cost}, initial model: {model}")

    model = batch_gradient_descent(housing_dataset.X_train, housing_dataset.y_train, model, cost_function,
                                   learning_rate=learning_rate, max_iter=1000)

    final_cost = cost_function(housing_dataset.X_train, housing_dataset.y_train, model)
    print(f"Final cost: {final_cost}, final model: {model}")

def main():
    housing_dataset = load(norm=False)
    print("Run with small learning rate:")
    run(housing_dataset, learning_rate=1e-9)
    print("Run with TOO large learning rate:")
    run(housing_dataset, learning_rate=1e-6)
    print("Run with fine-tuned learning rate:")
    run(housing_dataset, learning_rate=9e-7)

    print("Run NORMALIZED with reasonable learning rate")
    normalized_housing_dataset = load(norm=True)
    run(normalized_housing_dataset, learning_rate=0.1)
    run(normalized_housing_dataset, learning_rate=0.03)

    print("Run NORMALIZED with too small learning rate")
    run(normalized_housing_dataset, learning_rate=0.0001)
    print("Run NORMALIZED with too large learning rate")
    run(normalized_housing_dataset, learning_rate=0.97)


if __name__ == "__main__":
    main()
