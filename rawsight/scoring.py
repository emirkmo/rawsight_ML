from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

TruePositive: TypeAlias = int
TrueNegative: TypeAlias = int
FalseNegative: TypeAlias = int
FalsePositive: TypeAlias = int


@dataclass
class ConfusionMatrix:
    tp: TruePositive
    tn: TrueNegative
    fn: FalseNegative
    fp: FalsePositive


def accuracy(fy, y) -> float:
    return np.mean(fy == y)  # type: ignore


def precision(tp: TruePositive, fp: FalsePositive) -> float:
    if fp == 0 and tp == 0:
        return 0
    """Precision is the ratio of true positives to the sum of true and false positives."""
    return tp / (tp + fp)


def recall(tp: TruePositive, fn: FalseNegative) -> float:
    """Recall is the ratio of true positives to the sum of true positives and false negatives."""
    return tp / (tp + fn)


def f1_score(precision: float, recall: float) -> float:
    """F1 score is the harmonic mean of precision and recall."""
    return 2 * (precision * recall) / (precision + recall)


def _get_pos_neg(y: NDArray[np._IntType]) -> tuple[set[int], set[int]]:
    pos = set(np.flatnonzero(y.astype(bool)))
    neg = set(np.flatnonzero(~y.astype(bool)))
    return pos, neg


def confusion_matrix(
    y_pred: NDArray[np._IntType], y_actual: NDArray[np._IntType]
) -> ConfusionMatrix:
    ppos, pneg = _get_pos_neg(y_pred)
    apos, aneg = _get_pos_neg(y_actual)

    tp = len(ppos & apos)
    tn = len(pneg & aneg)
    fp = len(ppos & aneg)
    fn = len(pneg & apos)

    return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)


def calc_f1(matrix: ConfusionMatrix) -> float:
    rec = recall(matrix.tp, matrix.fn)
    prec = precision(matrix.tp, matrix.fp)
    if prec == 0 and rec == 0:
        return 0
    return f1_score(prec, rec)


def select_threshold(
    y: NDArray[np._IntType], p_val: NDArray[np._FloatType]
) -> tuple[float, float]:
    """Calculate threshold with highest F1 score, returning both."""
    best_epsilon = 0
    best_f1 = 0
    current_f1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        y_pred: NDArray[np._IntType] = p_val < epsilon
        cmatrix = confusion_matrix(y_pred, y)
        current_f1 = calc_f1(cmatrix)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epsilon = epsilon
    return best_epsilon, best_f1
