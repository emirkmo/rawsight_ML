import pandas as pd
from datasets import Dataset
from utils.normalization import MaxNorm
import numpy as np
from utils.regression import run_regression
from utils.cost_functions import least_squares_cost_function

def test_bb():
    df = pd.read_csv('../datasets/blackbody_10000_1e50_Hz.csv')
    df.drop(['droppable'], inplace=True, axis=1)
    df = df[df.Flux >= 1e-1]  # remove near-zero values
    df['nu2'] = -3*np.log(df['nu'])
    df['Flux'] = np.log(1./df['Flux'])
    bb = Dataset(name="Blackbody", df=df, target_name="Flux")
    bb.split_train_test(test_size=0.1)
    bb.normalize_features(normalizer=MaxNorm())

    # We don't have to normalize since we work in natural log space.
    #target_normalizer = MaxNorm()
    #y_norm = target_normalizer(bb.y_train)

    print(bb.X_train.shape)
    lr = run_regression("linear", bb.X_train, bb.y_train, learning_rate=0.05, max_iter=100000, w=(-100., +100.), b=-100.,
                        regularization_param=0.01)

    model = lr.model
    print(model.parameters)
    print(model.parameters[0]/bb.normalizer.norm['max'])
    #bb.denormalize_features()
    y_pred = lr.predict(bb.X_test)
    print(lr.score())
    print(lr.score(bb.X_test, bb.y_test))
    print(least_squares_cost_function(bb.X_test, bb.y_test, model))
    assert True
    return model, lr, bb, y_pred


def transform(w1, w2, b, y, x1_norm, x2_norm):
    from astropy.constants import c, h, k_B
    y = np.exp(1/y)
    w1 = w1/x1_norm
    w2 = w2/x2_norm
    A = c**2 * np.exp(-b) / (2 * h)
    T = (h/k_B/w1).cgs
    return y, T, A.cgs, w1, w2


if __name__ == "__main__":
    print("------")
    model, lr, bb, yp = test_bb()
    print("------")
    y, T, A, w1, w2 = transform(
        model.parameters[0][0], model.parameters[0][1],
        model.parameters[1], yp,
        bb.normalizer.norm['max'][0], bb.normalizer.norm['max'][1]
    )
    print(y,T,A,w1,w2)
    print("------")
