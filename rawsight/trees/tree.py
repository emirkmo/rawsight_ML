from typing import TypeAlias

TreeIndices: TypeAlias = list[int]  # Indices of the samples in the node.
ChildNodes: TypeAlias = tuple[
    TreeIndices, ...
]  # Indices of the samples in the N child nodes
TreeNode: TypeAlias = tuple[
    ChildNodes, int, float
]  # ChildNode branches, feature index, and information gain.
Tree: TypeAlias = list[TreeNode]  # list of TreeNodes.
