import numpy as np


def accuracy(fy, y):
    return np.mean(fy == y)
