import scipy.io as sio
import numpy as np

from sklearn import linear_model


def read_data():
    data = sio.loadmat("../octave/ex3/ex3data1.mat")
    x = data['X']
    y = data['y'].flatten()

    return (x, y)


X, y = read_data()
# C is inverse of lambda (I think) so 10 is same as 0.1 in non scipy implementation
regression = linear_model.LogisticRegression(C=10)
print("Fitting...")
regression.fit(X, y)

accuracy = regression.score(X, y) * 100
print('')
print('Train Accuracy: {}'.format(accuracy))
