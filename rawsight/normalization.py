"""
Data normalizers. Normalizers
follow a common NormalizerProtocol interface.
"""
import numpy as np
from typing import Protocol, Optional
from numpy.typing import ArrayLike


class NormalizerProtocol(Protocol):
    """Protocol for normalization defining the interface."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...

    def normalize(self, x: np.ndarray) -> np.ndarray:
        ...

    # noinspection PyPropertyDefinition
    @property
    def norm(self) -> dict[str, Optional[float | ArrayLike]]:
        ...

    def inverse(self, x: np.ndarray) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        ...

    def apply_norm(self, x: np.ndarray) -> np.ndarray:
        ...


class ZScoreNorm:
    """z-score normalization: subtract mean, divide by
    the standard deviation of the data. The normalization
    parameters are stored in the `norm` dictionary."""

    def __init__(self):
        self._norm_pars = {"mean": None, "std": None}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.normalize(x)

    @staticmethod
    def _norm_function(x: np.ndarray, mean: ArrayLike, std: ArrayLike) -> np.ndarray:
        """Normalization function."""
        return (x - mean) / std

    def normalize(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=0)
        self._norm_pars["mean"], self._norm_pars["std"] = mean, std
        return self._norm_function(x, mean, std)

    @property
    def norm(self) -> dict[str, Optional[float | ArrayLike]]:
        return self._norm_pars

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        if list(self.norm.values())[0] is None:
            raise ValueError("Normalization parameters are not set.")
        return x * self.norm["std"] + self.norm["mean"]

    def __repr__(self):
        return f"ZScoreNorm() with parameters: {self.norm}"

    def apply_norm(self, x: np.ndarray) -> np.ndarray:
        if list(self.norm.values())[0] is None:
            raise ValueError("Normalization parameters are not set.")
        return self._norm_function(x, self.norm["mean"], self.norm["std"])


class MaxNorm:
    """Max normalization: divide by
    the maximum value of the data. The normalization
    parameters are stored in the `norm` dictionary."""

    def __init__(self):
        self._norm_pars = {"max": None}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.normalize(x)

    @staticmethod
    def _norm_function(x: np.ndarray, max: ArrayLike) -> np.ndarray:
        """Normalization function."""
        return x / max

    def normalize(self, x: np.ndarray) -> np.ndarray:
        max = x.max(axis=0)
        self._norm_pars["max"] = max
        return self._norm_function(x, max)

    @property
    def norm(self) -> dict[str, Optional[float | ArrayLike]]:
        return self._norm_pars

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        if list(self.norm.values())[0] is None:
            raise ValueError("Normalization parameters are not set.")
        return x * self.norm["max"]

    def __repr__(self):
        return f"MaxNorm() with parameters: {self.norm}"

    def apply_norm(self, x: np.ndarray) -> np.ndarray:
        if list(self.norm.values())[0] is None:
            raise ValueError("Normalization parameters are not set.")
        return self._norm_function(x, self.norm["max"])


class MeanNorm:
    """Mean normalization: subtract mean, divide by
    the range of the data. The normalization
    parameters are stored in the `norm` dictionary."""

    def __init__(self):
        self._norm_pars = {"mean": None, "data_range": None}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.normalize(x)

    @staticmethod
    def _norm_function(
        x: np.ndarray, mean: ArrayLike, data_range: ArrayLike
    ) -> np.ndarray:
        """Normalization function."""
        return (x - mean) / data_range

    def normalize(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=0)
        data_range = x.max(axis=0) - x.min(axis=0)
        self._norm_pars["mean"] = mean
        self._norm_pars["data_range"] = data_range
        return self._norm_function(x, mean, data_range)

    @property
    def norm(self) -> dict[str, Optional[float | ArrayLike]]:
        return self._norm_pars

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        if list(self.norm.values())[0] is None:
            raise ValueError("Normalization parameters are not set.")
        return x * self.norm["data_range"] + self.norm["mean"]

    def __repr__(self):
        return f"MeanNorm() with parameters: {self.norm}"

    def apply_norm(self, x: np.ndarray) -> np.ndarray:
        if list(self.norm.values())[0] is None:
            raise ValueError("Normalization parameters are not set.")
        return self._norm_function(x, self.norm["mean"], self.norm["data_range"])
