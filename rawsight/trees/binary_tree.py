from typing import Any

import numpy as np

from ..cost_functions import NDArrayInt, binary_entropy_cost
from .infogain import InfoGain, info_gain_factory
from .splitting import binary_split
from .tree import Tree, TreeIndices
from .tree_builder import BuildPars, TreeBuilder

# def _binary_split(
#     data: np.ndarray, node_indices: TreeIndices, feature: int, **kwargs: Any
# ) -> tuple[TreeIndices, TreeIndices]:
#     """
#     Splits the data at the given node into
#     positive and negative branches

#     Args:
#         X (ndarray):             Data matrix of shape(n_samples, n_features)
#         node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
#         feature (int):           Index of feature to split on

#     Returns:
#         post_idx (list):     Indices with feature value == 1
#         neg_idx (list):    Indices with feature value == 0
#     """
#     if len(node_indices) == 0:
#         return [], []

#     rows = data[node_indices]
#     col = rows[:, feature]

#     pos = np.argwhere(col == 1).flatten()
#     neg = np.argwhere(col == 0).flatten()

#     pos_idx: TreeIndices = []
#     neg_idx: TreeIndices = []

#     if len(neg) > 0:
#         neg_idx = neg.tolist()
#     if len(pos) > 0:
#         pos_idx = pos.tolist()

#     pos_idx = np.array(node_indices)[pos_idx].tolist()
#     neg_idx = np.array(node_indices)[neg_idx].tolist()

#     return pos_idx, neg_idx


def _calc_binary_info_weights(
    y_node: NDArrayInt | TreeIndices, y_part: NDArrayInt | TreeIndices
) -> float:
    n_node = len(y_node)
    n_part = len(y_part)
    if n_node == 0 or n_part == 0:
        return 0.0
    w_part = n_part / n_node
    return w_part


def _binary_info_gain(
    y_node: NDArrayInt, y_left: NDArrayInt, y_right: NDArrayInt
) -> float:
    w_left = _calc_binary_info_weights(y_node, y_left)
    w_right = _calc_binary_info_weights(y_node, y_right)

    h_root = binary_entropy_cost(y_node)
    h_left, h_right = binary_entropy_cost(y_left), binary_entropy_cost(y_right)

    info_gain = h_root - (w_left * h_left + w_right * h_right)
    return info_gain


def calc_binary_info_gain(y_node: NDArrayInt, leafs: tuple[NDArrayInt, ...]) -> float:
    return _binary_info_gain(y_node, leafs[0], leafs[1])


binary_info_gain: InfoGain = info_gain_factory(
    "BinaryInfoGain", binary_split, calc_binary_info_gain
)


binary_tree_builder = TreeBuilder(binary_info_gain, pars=BuildPars())


def build_binary_tree(
    data: np.ndarray,
    y: NDArrayInt,
    max_depth: int = 2,
    min_info_gain: float = 0.02,
    **kwargs: Any
) -> Tree:
    binary_tree_builder.pars.max_depth = max_depth
    binary_tree_builder.pars.min_info_gain = min_info_gain
    return binary_tree_builder(data, y, **kwargs)
