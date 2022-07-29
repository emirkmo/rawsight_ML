from typing import Protocol
from numpy.typing import NDArray, ArrayLike


class Model(Protocol):
    def __call__(self, x: NDArray) -> NDArray:
        ...

    def partial_derivatives(self, x: NDArray) -> NDArray:
        ...

    def evaluate(self, x: NDArray, y: NDArray) -> ArrayLike | float:
        ...