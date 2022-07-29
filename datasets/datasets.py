from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

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
    target: pd.Series = None
    X_train: pd.DataFrame = NDArray
    X_test: pd.DataFrame = NDArray
    y_train: pd.Series = NDArray
    y_test: pd.Series = NDArray

    def __post_init__(self):
        """
        Initializes the dataframes for training, testing, and validation.
        """
        if isinstance(self.target_name, int):
            self.target_name = self.df.columns[self.target_name]
        self.target = self.df[self.target_name]
        self.features = self.df.drop(self.target_name, axis=1)

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


def load_housing_data(filename: str = "houses.csv") -> Dataset:
    """
    Loads the housing data from the given filename.
    """
    df = pd.read_csv(DATA_PATH/filename)
    dataset = Dataset(name="Housing", df=df, target_name="Price (1000s dollars)")
    dataset.split_train_test(test_size=0)
    return dataset
