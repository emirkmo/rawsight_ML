from typing import Optional

from .layers import DenseLayer, Layer
import numpy as np
from utils.cost_functions import CostFunction
from utils.optimizers import Optimizer


class Sequential:

    def __init__(self, layers: list[Layer], loss: CostFunction, optimizer: Optimizer,
                 optimizer_hyperparams: Optional[dict] = None):
        self.layers = layers
        self.costfunction = loss
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_hyperparams if optimizer_hyperparams is not None else {}

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1, verbose: bool = False) -> None:
        """
        Fit the model to the data.
        :param x: Data.
        :param y: Target values.
        :param epochs: Number of epochs to run.
        :param verbose: Whether to print progress.
        :return: self.
        """

        # run gradient descent
        x_init = x.copy()
        for epoch in range(epochs):
            for j, layer in enumerate(self.layers[::-1]):
                n_layers = len(self.layers)
                x = x_init
                for i in range(n_layers - j):
                    #layer.backward(self.layers[n_layers - i - 1].delta)
                    x = self.layers[i].forward(x)
                layer.model = self.optimizer(x, y, layer.model, self.costfunction, **self.optimizer_kwargs)
                layer.w = layer.model.parameters[0]
                layer.b = layer.model.parameters[1]

            if verbose:
                print("Epoch {}".format(epoch))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output of the model.
        :param x: Data.
        :return: Predicted values.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
