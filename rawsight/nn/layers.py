"""Class for Dense layer in a neural network."""
from typing import Optional, Protocol
from numpy.typing import ArrayLike, NDArray
from rawsight.models.logistic import LogisticNNModel
from rawsight.models.linear import LinearNNModel
from enum import Enum
import numpy as np
from rawsight.models.model import BaseLinearModel, BaseNeuralNetLinearModel


class Activation(Enum):
    """Enum for activation functions."""

    linear = "linear"
    sigmoid = "sigmoid"

    def get_function(self) -> type[BaseNeuralNetLinearModel]:
        """Get the activation function."""
        if self == self.linear:
            return LinearNNModel
        elif self == self.sigmoid:
            return LogisticNNModel
        else:
            raise ValueError("Invalid activation function.")


class Layer(Protocol):
    model: Optional[BaseNeuralNetLinearModel]
    w: Optional[ArrayLike]
    b: Optional[float]

    def __call__(self, *args, **kwargs) -> NDArray:
        raise NotImplementedError()

    def forward(self, x: NDArray) -> NDArray:
        """Forward pass of the layer."""
        raise NotImplementedError()

    def initialize_params(self) -> None:
        """Initialize the parameters of the layer."""
        raise NotImplementedError()

    def init_model(self) -> None:
        """Initialize the activation function model of the layer."""
        raise NotImplementedError()

    # def backward(self, *args, **kwargs) -> NDArray:
    #     """Backward pass of the layer."""
    #     raise NotImplementedError()


class DenseLayer:
    # model = None
    # w = None
    # b = None

    def __init__(self, neurons: int, input_dim: int, activation: str, name: str):
        """
        Initialize a DenseLayer.
        :param neurons: Number of units in the layer.
        :param input_dim: Number of input features.
        :param activation: Activation function.
        :param name: Name of the layer.
        """
        self.units = neurons
        self.input_dim = input_dim
        self.activation: Activation = Activation(activation)
        self.name = name
        self.w: Optional[NDArray] = None
        self.b: Optional[NDArray] = None
        self.params = None
        self.initialize_params()
        self.model: Optional[BaseLinearModel] = None
        self.init_model()

    def init_model(self) -> None:
        self.model = self.activation.get_function()(
            self.w, self.b, n_features=self.input_dim
        )

    def initialize_params(self) -> None:
        """Initialize the weights and biases."""
        self.w = np.random.randn(self.input_dim, self.units)
        self.b = np.random.randn(self.units, 1)
        self.params = [self.w, self.b]

    def forward(self, x: NDArray) -> NDArray:
        """
        Forward pass of the layer.
        :param x: Input to the layer.
        :return: Output of the layer.
        """
        return self.model(x)

    def __call__(self, *args: NDArray, **kwargs: ArrayLike | float) -> NDArray:
        """
        Call the layer.
        :param args: Input to the layer.
        :param kwargs: Keyword arguments.
        :return: Output of the layer.
        """
        return self.forward(*args, **kwargs)


def test_dense_layer(layer: Layer):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = layer(x)
    print(y)


if __name__ == "__main__":
    layer = DenseLayer(neurons=2, input_dim=3, activation="sigmoid", name="test")
    test_dense_layer(layer)
