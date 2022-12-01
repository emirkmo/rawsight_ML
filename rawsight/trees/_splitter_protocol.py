from typing import Any, Protocol

import numpy as np

from .tree import ChildNodes, TreeIndices


class Splitter(Protocol):
    def __call__(
        self, data: np.ndarray, node_indices: TreeIndices, feature: int, **kwargs: Any
    ) -> ChildNodes:
        ...
