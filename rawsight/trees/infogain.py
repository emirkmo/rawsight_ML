import numpy as np

from rawsight.cost_functions import NDArrayInt, binary_entropy_cost

from .splitting import binary_split


def _calc_binary_info_weights(
    y_node: NDArrayInt | list[int], y_part: NDArrayInt | list[int]
) -> float:
    w_part = len(y_part) / len(y_node)
    return w_part


def info_gain(y_node: NDArrayInt, y_left: NDArrayInt, y_right: NDArrayInt) -> float:
    w_left = _calc_binary_info_weights(y_node, y_left)
    w_right = _calc_binary_info_weights(y_node, y_right)

    h_root = binary_entropy_cost(y_node)
    h_left, h_right = binary_entropy_cost(y_left), binary_entropy_cost(y_right)

    info_gain = h_root - (w_left * h_left + w_right * h_right)
    return info_gain


def calc_binary_info_gain(
    data: np.ndarray, y: NDArrayInt, node_indices: list[int], feature: int
) -> float:
    pos_indices, neg_indices = binary_split(data, node_indices, feature)
    y_node = y[node_indices]
    y_pos = y[pos_indices]
    y_neg = y[neg_indices]
    return info_gain(y_node, y_pos, y_neg)
