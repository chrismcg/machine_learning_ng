import scipy.io as sio
import numpy as np

from sklearn.neural_network import MLPClassifier

def read_data():
    data = sio.loadmat("../octave/ex4/ex4data1.mat")
    x = data['X']
    y = data['y'][:, 0]

    return (x, y)

X, y = read_data()

# clf = MLPClassifier(activation='logistic', max_iter=50, solver='lbfgs', hidden_layer_sizes=(25), verbose=True)
clf = MLPClassifier(verbose=True)
print("Fitting...")
clf.fit(X, y)

accuracy = clf.score(X, y) * 100
print('')
print('Train Accuracy: {}'.format(accuracy))
