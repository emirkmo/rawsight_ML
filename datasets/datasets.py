from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field
from utils.normalization import NormalizerProtocol, ZScoreNorm
from utils.input_validation import get_n_features

import pandas as pd
from numpy.typing import NDArray

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
    y_train: NDArray = None
    y_test: NDArray = None
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
            self.X_train = (self.features.iloc[:int(len(self.features) * test_size)]).values
            self.y_train = (self.target.iloc[:int(len(self.target) * test_size)]).values

            self.X_test = (self.features.iloc[int(len(self.features) * test_size):]).values
            self.y_test = (self.target.iloc[int(len(self.target) * test_size):]).values

    def normalize_features(self, normalizer: Optional[NormalizerProtocol] = None) -> None:
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
    df = pd.read_csv(DATA_PATH/filename)
    dataset = Dataset(name="Housing", df=df, target_name="Price (1000s dollars)")
    dataset.split_train_test(test_size=0)
    return dataset


def load_tumor_simple(filename: str = "tumors_simple.csv") -> Dataset:
    """
    Loads the tumor data from the given filename.
    """
    df = pd.read_csv(DATA_PATH/filename)
    dataset = Dataset(name="Tumor", df=df, target_name="is_tumor")
    dataset.split_train_test(test_size=0)
    return dataset


def load_coffee_data(filename: str = "coffee_data.csv") -> Dataset:
    """
    Loads the coffee data from the given filename.
    """
    df = pd.read_csv(DATA_PATH/filename)
    dataset = Dataset(name="Coffee", df=df, target_name="Taste")
    dataset.split_train_test(test_size=0)
    return dataset