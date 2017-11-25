import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

def read_data():
    df = pd.read_csv("../octave/ex2/ex2data2.txt", names=['Test1', 'Test2', 'Accepted'])

    x = df.as_matrix(["Test1", "Test2"])
    y = df["Accepted"]

    return (x, y)


def plot_data(x, y):
    positives = x[(y == 1)]
    negatives = x[(y == 0)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(positives[:, 0], positives[:, 1], 'k+', linewidth=2, markersize=7, label="Accepted")
    ax.plot(negatives[:, 0], negatives[:, 1], 'ko', markerfacecolor='y', markersize=7, label="Not accepted")
    ax.legend()
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")

    return ax


def plot_decision_boundary(regression, poly, x, y):
    ax = plot_data(x[:, 1:3], y)

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.size, v.size))
    for i in range(u.size):
        for j in range(v.size):
            input = poly.fit_transform(np.array([[u[i], v[j]]]))
            value = regression.decision_function(input)[0]
            z[i, j] = value

    z = z.transpose()
    ax.contour(u, v, z, 0, linewidths=2)
    ax.legend()

X, y = read_data()
poly = PolynomialFeatures(6)
X = poly.fit_transform(X)
regression = linear_model.LogisticRegression(verbose=1)
regression.fit(X, y)

accuracy = regression.score(X, y) * 100
print('')
print('Train Accuracy: {}'.format(accuracy))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')

plot_decision_boundary(regression, poly, X, y)
plt.show()
