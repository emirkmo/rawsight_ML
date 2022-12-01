import numpy as np
import pandas as pd
import pytest

from rawsight.trees import (
    ChildNodes,
    Tree,
    TreeIndices,
    TreeNode,
    binary_split,
    build_binary_tree,
)
from rawsight.trees._splitter_protocol import Splitter
from rawsight.trees.binary_tree import binary_info_gain
from rawsight.trees.infogain import BaseInfoGain, InfoGain, InfoGainType

X_train = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]
)
y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
ref_tree: Tree = [
    (([0, 1, 4, 5, 7], [2, 3, 6, 8, 9]), 2, 0.2780719051126377),
    (([0, 1, 4, 7], [5]), 0, 0.7219280948873623),
    (([8], [2, 3, 6, 9]), 1, 0.7219280948873623),
]


def test_info_gain_factory():
    binsplit = binary_info_gain.splitter(X_train, list(range(len(X_train))), 1)
    assert binsplit == ([0, 4, 5, 8], [1, 2, 3, 6, 7, 9])
    infogained = binary_info_gain.info_gainer(
        y_train, (y_train[binsplit[0]], y_train[binsplit[1]])
    )

    pytest.approx(infogained, 0.12451124978365313)
    assert isinstance(binary_info_gain, InfoGain)
    # print(type())
    assert binary_info_gain.__class__.__name__ == "BinaryInfoGain"


def test_binary_tree():
    bintree: Tree = build_binary_tree(
        X_train, y_train, max_depth=2, min_info_gain=0.001
    )

    for i, node in enumerate(bintree):
        childnodes = node[0]
        assert childnodes == ref_tree[i][0]

        assert node[1] == ref_tree[i][1]

        pytest.approx(node[2], ref_tree[i][2], abs=1e-3)

    print(bintree)
    print(ref_tree)
