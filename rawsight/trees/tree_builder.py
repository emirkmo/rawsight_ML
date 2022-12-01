from dataclasses import dataclass
from typing import Any, Optional, TypeGuard

import numpy as np

from ..cost_functions import NDArrayInt
from .infogain import InfoGain
from .splitting import get_best_split
from .tree import ChildNodes, Tree, TreeIndices, TreeNode


@dataclass
class BuildPars:
    branches: int = 0
    max_depth: int = 2
    current_depth: int = 0  # equivalent to starting depth before use.
    min_info_gain: float = (
        0.02  # if info gain is less than min_info_gain, then we don't split.
    )

    def dive(self, new_depth: int) -> None:
        if self.current_depth < new_depth:
            self.current_depth = new_depth

    def branch(self, new_branches: int) -> None:
        self.branches += new_branches


class TreeBuilder:
    def __init__(
        self,
        infogain: InfoGain,
        input_tree: Optional[Tree] = None,
        pars: BuildPars = BuildPars(),
    ) -> None:
        if input_tree is None:
            input_tree = []
        self.tree = input_tree
        self.pars = pars
        self.infogain = infogain

    def build_child_nodes(
        self,
        data: np.ndarray,
        y: NDArrayInt,
        node_indices: TreeIndices,
        current_depth: int = 0,
        **kwargs: Any
    ) -> None:
        # Stop if we have reached the maximum depth.
        if current_depth >= self.pars.max_depth:
            return

        # Pick best feature and split node into N leafs.
        leafs, feature, info_gain = self.split(data, y, node_indices, **kwargs)
        self.tree.append((leafs, feature, info_gain))

        # These should be moved to a Tree class:
        self.pars.branch(len(leafs))
        self.pars.dive(current_depth)

        # Stop if we have reached the minimum information gain.
        if info_gain < self.pars.min_info_gain:
            return

        # Recursively build child nodes.
        for node in leafs:
            self.build_child_nodes(
                data, y, node, current_depth=current_depth + 1, **kwargs
            )

    def build_recursive(
        self,
        data: np.ndarray,
        y: NDArrayInt,
        node_indices: Optional[TreeIndices] = None,
        **kwargs: Any
    ) -> None:
        """Build a tree recursively starting from pars.current_depth."""
        if node_indices is None:
            node_indices = list(range(len(y)))

        self.build_child_nodes(data, y, node_indices, self.pars.current_depth, **kwargs)

    def split(
        self, data: np.ndarray, y: NDArrayInt, node_indices: TreeIndices, **kwargs: Any
    ) -> TreeNode:
        best_feature, best_info_gain = get_best_split(
            self.infogain, data, y, node_indices
        )
        leafs: ChildNodes = self.infogain.splitter(
            data, node_indices, best_feature, **kwargs
        )
        return leafs, best_feature, best_info_gain

    def __call__(
        self,
        data: np.ndarray,
        y: NDArrayInt,
        node_indices: Optional[TreeIndices] = None,
        **kwargs: Any
    ) -> Tree:
        self.build_recursive(data, y, node_indices, **kwargs)
        return self.tree
