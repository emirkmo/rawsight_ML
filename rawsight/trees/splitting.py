import numpy as np


def binary_split(
    data: np.ndarray, node_indices: list[int], feature: int
) -> tuple[list[int], list[int]]:
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

    pos_idx: list[int] = []
    neg_idx: list[int] = []

    if len(neg) > 0:
        neg_idx = neg.tolist()
    if len(pos) > 0:
        pos_idx = pos.tolist()

    pos_idx = np.array(node_indices)[pos_idx].tolist()
    neg_idx = np.array(node_indices)[neg_idx].tolist()

    return pos_idx, neg_idx
