from typing import Any

import numpy as np

from ..cost_functions import NDArrayInt
from .infogain import InfoGain
from .tree import TreeIndices


def get_best_split(
    infogain: InfoGain,
    data: np.ndarray,
    y: NDArrayInt,
    node_indices: TreeIndices,
    **kwargs: Any
) -> tuple[int, float]:
    """
    Finds the best feature to split on and its info gain for the given node

    Args:
        data (ndarray):             Data matrix of shape(n_samples, n_features)
        y (ArrayLike):             Target vector of shape(n_samples,)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.

    Returns:
        best_feature (int):     Index of the best feature to split on
        best_info_gain (int):   Information gain from the best split
    """
    best_feature = -1
    best_info_gain = 0
    for feature in range(data.shape[1]):
        info_gain = infogain(data, y, node_indices, feature, **kwargs)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature

    return best_feature, best_info_gain


def binary_split(
    data: np.ndarray, node_indices: TreeIndices, feature: int, **kwargs: Any
) -> tuple[TreeIndices, TreeIndices]:
    """
    Splits the data at the given node into
    positive and negative branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        post_idx (list):     Indices with feature value == 1
        neg_idx (list):    Indices with feature value == 0
    """
    if len(node_indices) == 0:
        return [], []

    rows = data[node_indices]
    col = rows[:, feature]

    pos = np.argwhere(col == 1).flatten()
    neg = np.argwhere(col == 0).flatten()

    pos_idx: TreeIndices = []
    neg_idx: TreeIndices = []

    if len(neg) > 0:
        neg_idx = neg.tolist()
    if len(pos) > 0:
        pos_idx = pos.tolist()

    pos_idx = np.array(node_indices)[pos_idx].tolist()
    neg_idx = np.array(node_indices)[neg_idx].tolist()

    return pos_idx, neg_idx
