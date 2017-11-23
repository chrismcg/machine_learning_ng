import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

df = pd.read_csv("../octave/ex2/ex2data1.txt", names=['Score1', 'Score2', 'Admitted'])

X = df.as_matrix(["Score1", "Score2"])
y = df.as_matrix(["Admitted"])


def plot_data(x, y):
    positives = x[(y == 1)[:, 0]]
    negatives = x[(y == 0)[:, 0]]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(positives[:, 0], positives[:, 1], 'k+', linewidth=2, markersize=7, label="Admitted")
    ax.plot(negatives[:, 0], negatives[:, 1], 'ko', markerfacecolor='y', markersize=7, label="Not admitted")
    ax.legend()
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")

    return ax

def plot_decision_boundary(theta, x, y):
    ax = plot_data(x[:, 1:3], y)

    plot_x = np.array([x[:, 1].min() - 2, x[:, 2].max() + 2])
    plot_y = (-1 / theta[2]) * ((plot_x * theta[1]) + theta[0])
    ax.plot(plot_x, plot_y, label="Decision Boundary")
    ax.legend()
    ax.axis([30, 100, 30, 100])


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def cost_function(theta, x, y):
    m = y.size
    # ensure it's a vector not an array for dot with reshape below
    h = sigmoid(X.dot(theta.reshape((theta.size, 1))))
    y_transpose = y.transpose()
    gradients = np.zeros(theta.size)

    j = (-y_transpose.dot(np.log(h)) - ((1 - y_transpose).dot(np.log(1 - h)))) / m

    for i in range(theta.size):
        gradients[i] = (h - y).transpose().dot(x[:, i]) / m

    return (j, gradients)


def predict(theta, x):
    return sigmoid(X.dot(theta)) > 0.5


plot_data(X, y)
plt.show()

m, n = X.shape

X = np.c_[np.ones(m), X]
initial_theta = np.zeros((n + 1, 1))

cost, gradients = cost_function(initial_theta, X, y)

print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): ')
print(' {} '.format(gradients))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')

initial_theta = np.zeros((n + 1, 1))
result = op.minimize(lambda t: cost_function(t, X, y), initial_theta, method="Newton-CG", jac=True, options={'maxiter': 400, 'disp': True})
cost = result.fun
theta = result.x

print('Cost at theta found by Newton-CG: {}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta: ')
print(' {} '.format(theta))
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201')

plot_decision_boundary(theta, X, y)
plt.show()

probability = sigmoid(np.array([1, 45, 85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of {}'.format(probability))
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)

accuracy = (p == y[:, 0]).mean() * 100
print('Train Accuracy: {}'.format(accuracy))
print('Expected accuracy (approx): 89.0')


