from numpy.typing import NDArray


def get_n_features(x: NDArray) -> float:
    if len(x.shape) == 1:
        return 1
    return x.shape[1]
