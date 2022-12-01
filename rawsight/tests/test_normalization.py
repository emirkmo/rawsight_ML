import pytest
import numpy as np
from rawsight.datasets import load_housing_data
from rawsight import ZScoreNorm, MaxNorm, MeanNorm
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def housing_data():
    yield load_housing_data()


def test_housing_dataset_init(housing_data):
    # test inititalization
    p2p_init = np.array([2.406e03, 4.000e00, 1.000e00, 9.500e01])
    assert np.ptp(housing_data.X_train, axis=0) == pytest.approx(p2p_init, abs=1e-2)


def test_zscorenorm_vs_sklearn(housing_data):
    # Ensure my output is in the same format as the sklearn output.
    xnorm_mine = ZScoreNorm().normalize(housing_data.X_train)
    xnorm_sk = StandardScaler().fit_transform(housing_data.X_train)
    assert np.all(xnorm_sk[:5] == pytest.approx(xnorm_mine[:5]))


@pytest.mark.parametrize("normalizer", [ZScoreNorm(), MaxNorm(), MeanNorm()])
def test_round_trip(housing_data, normalizer):
    p2p_init = np.ptp(housing_data.X_train, axis=0)
    housing_data.normalize_features(normalizer=normalizer)
    p2p_norm = np.ptp(housing_data.X_train, axis=0)

    # Ensure we round trip:
    housing_data.denormalize_features()
    assert np.ptp(housing_data.X_train, axis=0) == pytest.approx(p2p_init, abs=1e-1)
    housing_data.normalize_features(normalizer=normalizer)
    assert np.ptp(housing_data.X_train, axis=0) == pytest.approx(p2p_norm, abs=1e-2)


def test_maxnorm():
    a = np.array([1.0, 2.0, 1.0])
    normalizer = MaxNorm()
    a_norm = normalizer.normalize(a)
    assert a_norm == pytest.approx(np.array([0.5, 1.0, 0.5]))
    assert np.max(a_norm, axis=0) == pytest.approx(1.0)
    assert np.max(a, axis=0) > np.max(a_norm, axis=0)
    assert normalizer.norm["max"] == np.max(a, axis=0)


def test_meannorm():
    a = np.array([1.0, 2.0, 1.0])
    normalizer = MeanNorm()
    a_norm = normalizer.normalize(a)
    assert normalizer.norm["mean"] == np.mean(a, axis=0)
    assert normalizer.norm["data_range"] == np.max(a, axis=0) - np.min(a, axis=0)
    assert np.mean(normalizer.norm["data_range"] * a_norm, axis=0) == pytest.approx(0)


if __name__ == "__main__":
    test_housing_dataset_init()
    test_zscorenorm_vs_sklearn()
    test_round_trip()
    test_maxnorm()
    test_meannorm()
