import types
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..cost_functions import NDArrayInt
from ._splitter_protocol import Splitter
from .tree import ChildNodes, TreeIndices


class InfoGainType(Protocol):
    def __call__(self, y_node: NDArrayInt, leafs: tuple[NDArrayInt, ...]) -> float:
        ...


def calc_info_gain(
    data: np.ndarray,
    y: NDArrayInt,
    node_indices: TreeIndices,
    feature: int,
    splitter: Splitter,
    info_gainer: InfoGainType,
    **kwargs: Any
) -> float:
    y_node: NDArrayInt = y[node_indices]

    # split node into n leafs, as determined by the splitter.
    split_indices: ChildNodes = splitter(data, node_indices, feature, **kwargs)
    leafs: tuple[NDArrayInt, ...] = tuple(y[ind] for ind in split_indices)

    return info_gainer(y_node, leafs)


@runtime_checkable
class InfoGain(Protocol):
    splitter: Splitter
    info_gainer: InfoGainType

    def __call__(
        self,
        data: np.ndarray,
        y: NDArrayInt,
        node_indices: TreeIndices,
        feature: int,
        **kwargs: Any
    ) -> float:
        raise NotImplementedError("InfoGain is an abstract class.")


class BaseInfoGain:
    splitter: Splitter
    info_gainer: InfoGainType

    def __call__(
        self,
        data: np.ndarray,
        y: NDArrayInt,
        node_indices: TreeIndices,
        feature: int,
        **kwargs: Any
    ) -> float:

        return calc_info_gain(
            data, y, node_indices, feature, self.splitter, self.info_gainer, **kwargs
        )


def info_gain_factory(
    name: str, splitter: Splitter, info_gainer: InfoGainType
) -> InfoGain:

    # kwds = {"splitter": types.MethodType(splitter), "info_gainer": info_gainer},
    # new_class = types(name, (InfoGain, BaseInfoGain,),)

    new_info_gain_class: type[BaseInfoGain] = type(
        name,
        (BaseInfoGain,),
        {"splitter": staticmethod(splitter), "info_gainer": staticmethod(info_gainer)},
    )
    initialized_info_gain_class: InfoGain = new_info_gain_class()
    return initialized_info_gain_class
