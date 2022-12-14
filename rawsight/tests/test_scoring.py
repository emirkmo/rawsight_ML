import numpy as np
import pytest

from rawsight.scoring import (
    ConfusionMatrix,
    _get_pos_neg,
    accuracy,
    calc_f1,
    confusion_matrix,
    f1_score,
    pdf_gaussian,
    precision,
    recall,
    select_threshold,
)

from .inputs import X, Y, mu, var


@pytest.fixture(scope="module")
def p_val():
    return pdf_gaussian(X, mu, var)


def test_pdf_gaussian(p_val):
    # p_val = pdf_gaussian(X, mu, var)
    assert p_val[0] > 0.04 and p_val[0] < 0.042
    assert max(p_val) == pytest.approx(0.08990852779269494)
    assert min(p_val) == pytest.approx(4.513250930309844e-36)


def test_precision():
    assert precision(2, 10) == 2 / 12
    assert precision(0, 0) == 0
    assert precision(0, 100000000) == 0


def test_recall():
    assert recall(2, 8) == 2 / 10
    assert recall(0, 0) == 0
    assert recall(0, 10000000000) == 0


def test_accuracy():
    assert accuracy(np.array([1, 1]), np.array([1, 1])) == 1
    assert accuracy(mu, var) == 0


def test_f1_score():
    f1 = f1_score(precision(2, 10), recall(2, 8))
    assert f1 == pytest.approx(0.1818181818181818)

    with pytest.raises(ZeroDivisionError):
        f1_score(0, 0)


def test_confusion_matrix(p_val):
    epsilon = 8.990852779269495e-05
    y_pred = p_val < epsilon
    cmatrix = confusion_matrix(y_pred, Y)
    assert isinstance(cmatrix, ConfusionMatrix)
    assert cmatrix.tp == 7
    assert cmatrix.fp == 0
    assert cmatrix.fn == 2
    assert cmatrix.tn == 298
    assert cmatrix.tp + cmatrix.fp + cmatrix.fn + cmatrix.tn == len(y_pred)


def test_ConfusionMatrix():
    cmatrix = ConfusionMatrix(tp=2, tn=298, fn=2, fp=0)
    assert isinstance(cmatrix, ConfusionMatrix)
    assert cmatrix.tp == 2
    assert cmatrix.fp == 0
    assert cmatrix.fn == 2
    assert cmatrix.tn == 298


def test_calc_f1(p_val):
    epsilon = 8.990852779269495e-05
    y_pred = p_val < epsilon
    cmatrix = confusion_matrix(y_pred, Y)
    f1 = calc_f1(cmatrix)
    prec = precision(cmatrix.tp, cmatrix.fp)
    rec = recall(cmatrix.tp, cmatrix.fn)

    assert pytest.approx(f1) == pytest.approx(f1_score(prec, rec))

    assert calc_f1(ConfusionMatrix(0, 1000, 0, 0)) == 0


def test_select_treshold(p_val):
    best_epsilon, best_f1 = select_threshold(Y, p_val)
    assert best_epsilon == pytest.approx(8.990852779269495e-05)
    assert best_f1 == pytest.approx(0.875)


def test_get_pos_neg(p_val):
    epsilon = 8.990852779269495e-05
    y_pred = p_val < epsilon
    pos, neg = _get_pos_neg(y_pred)

    assert len(pos) == 7
    assert len(neg) == 300
    assert len(pos) + len(neg) == len(y_pred)

    pos, neg = _get_pos_neg(np.array([0, 1]))
    assert pos == set([1])
    assert neg == set([0])
    assert len(pos) == len(neg)
