import tensorflow as tf
from numpy.typing import ArrayLike

from datasets import load_coffee_data
import numpy as np
from rawsight.nn.layers import DenseLayer

data = load_coffee_data()
data.normalize_features()

l1 = DenseLayer(neurons=3, input_dim=2, activation="sigmoid", name='l1')
l2 = DenseLayer(neurons=1, input_dim=2, activation="sigmoid", name='l2')


def my_sequential(x):
    W1_tmp = np.array([[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]])
    b1_tmp = np.array([-9.82, -9.28, 0.96])
    W2_tmp = np.array([[-31.18], [-27.59], [-32.56]])
    b2_tmp = np.array([15.41])
    l1.w = W1_tmp
    l1.b = b1_tmp
    l2.w = W2_tmp
    l2.b = b2_tmp
    l1.init_model()
    l2.init_model()
    a1 = l1(x)
    a2 = l2(a1)
    return a2


data.X_test = data.normalizer.apply_norm(np.array([[200, 13.9], [200, 17]]))
print(my_sequential(data.X_test))
print(f"expected:{np.array([[9.72e-01], [3.29e-08]])}")
print((my_sequential(data.X_test) >= 0.5).astype(int))

