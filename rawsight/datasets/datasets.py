from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from rawsight.input_validation import get_n_features
from rawsight.normalization import NormalizerProtocol, ZScoreNorm

DATA_PATH = Path(__file__).parent


@dataclass
class Dataset:
    """
    Dataclass for a dataset.
    """

    name: str
    df: pd.DataFrame
    target_name: Optional[str | int] = -1
    features: pd.DataFrame = None
    num_features: int = field(init=False)
    target: pd.Series = None
    X_train: NDArray = None
    X_test: NDArray = None
    X_cv: NDArray = None
    y_train: NDArray = None
    y_test: NDArray = None
    y_cv: NDArray = None
    normalizer: NormalizerProtocol = field(default_factory=ZScoreNorm)

    def __post_init__(self):
        """
        Initializes the dataframes for training, testing, and validation.
        """
        if isinstance(self.target_name, int):
            self.target_name = self.df.columns[self.target_name]
        self.target = self.df[self.target_name]
        self.features = self.df.drop(self.target_name, axis=1)
        self.num_features = get_n_features(self.features)

    def split_train_test(self, test_size: float = 0.2) -> None:
        """
        Splits the data into training and testing sets.
        """
        if test_size == 0:
            self.X_train = self.features.values
            self.y_train = self.target.values
        else:
            self.X_train = (
                self.features.iloc[: int(len(self.features) * test_size)]
            ).values
            self.y_train = (
                self.target.iloc[: int(len(self.target) * test_size)]
            ).values

            self.X_test = (
                self.features.iloc[int(len(self.features) * test_size) :]
            ).values
            self.y_test = (self.target.iloc[int(len(self.target) * test_size) :]).values

    def split_test_cv(self, cv_size: float = 0.5) -> None:
        """
        Splits the test set into test and cross-validation sets.
        """
        if cv_size == 0:
            self.X_cv = None
            self.y_cv = None
        if cv_size == 1:
            self.X_cv = self.X_test
            self.y_cv = self.y_test
        else:
            self.X_cv = self.X_test[int(len(self.X_test) * (1 - cv_size)) :]
            self.y_cv = self.y_test[int(len(self.y_test) * (1 - cv_size)) :]

            self.X_test = self.X_test[: int(len(self.X_test) * (1 - cv_size))]
            self.y_test = self.y_test[: int(len(self.y_test) * (1 - cv_size))]

    def normalize_features(
        self, normalizer: Optional[NormalizerProtocol] = None
    ) -> None:
        """
        Normalizes the input train and test samples of the dataset.
        Normalization factors are saved in the `norm` dictionary of the Normalizer.
        If None, the class instance normalizer is used, else its overwritten.
        """
        self.normalizer = normalizer or self.normalizer
        self.X_train = self.normalizer(self.X_train)
        if self.X_test is not None:
            self.X_test = self.normalizer(self.X_test)

    def denormalize_features(self) -> None:
        """
        Denormalizes the input train and test samples of the dataset.
        """
        self.X_train = self.normalizer.inverse(self.X_train)
        if self.X_test is not None:
            self.X_test = self.normalizer.inverse(self.X_test)


def load_housing_data(filename: str = "houses.csv") -> Dataset:
    """
    Loads the housing data from the given filename.
    """
    df = pd.read_csv(DATA_PATH / filename)
    dataset = Dataset(name="Housing", df=df, target_name="Price (1000s dollars)")
    dataset.split_train_test(test_size=0)
    return dataset


def load_tumor_simple(filename: str = "tumors_simple.csv") -> Dataset:
    """
    Loads the tumor data from the given filename.
    """
    df = pd.read_csv(DATA_PATH / filename)
    dataset = Dataset(name="Tumor", df=df, target_name="is_tumor")
    dataset.split_train_test(test_size=0)
    return dataset


def load_coffee_data(filename: str = "coffee_data.csv") -> Dataset:
    """
    Loads the coffee data from the given filename.
    """
    df = pd.read_csv(DATA_PATH / filename)
    dataset = Dataset(name="Coffee", df=df, target_name="Taste")
    dataset.split_train_test(test_size=0)
    return dataset


# Course 2 Week 3
def _gen_data(m: int, seed: int = 2, scale: float = 0.7):
    """generate a data set based on a x^2 with added noise"""
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal * (np.random.sample((m,)) - 0.5)
    x_ideal = x_train  # for redraw when new data included in X
    return x_train, y_train, x_ideal, y_ideal


def load_mnist(subset: bool = True) -> Dataset:
    # def load_data():
    X = np.load("X.npy")
    y = np.load("y.npy")
    df = pd.DataFrame({"X": X, "y": y})
    dataset = Dataset(name="MNIST", df=df, target_name="y")
    return dataset


def load_course2_week3_data() -> Dataset:
    x_train, y_train, x_ideal, y_ideal = _gen_data(40, 5, 0.7)
    df = pd.DataFrame({"X": x_train, "y": y_train})

    dataset = Dataset(name="C2W3", df=df, target_name="y")
    dataset.split_train_test(test_size=0.33)
    dataset.split_test_cv(cv_size=0.5)
    return dataset
