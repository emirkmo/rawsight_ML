from rawsight import least_squares_cost_function, MaxNorm, MeanNorm, ZScoreNorm
from datasets import load_housing_data, Dataset
from .multiple_linear_regression import run


def load(norm: bool = False):
    # Load and normalize the housing data
    housing_dataset = load_housing_data()
    if norm:
        housing_dataset.normalize_features(ZScoreNorm())
    return housing_dataset


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
