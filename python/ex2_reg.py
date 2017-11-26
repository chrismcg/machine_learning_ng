import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

df = pd.read_csv("../octave/ex2/ex2data2.txt", names=['Test1', 'Test2', 'Accepted'])

X = df.as_matrix(["Test1", "Test2"])
y = df.as_matrix(["Accepted"])


def plot_data(x, y):
    positives = x[(y == 1)[:, 0]]
    negatives = x[(y == 0)[:, 0]]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(positives[:, 0], positives[:, 1], 'k+', linewidth=2, markersize=7, label="Accepted")
    ax.plot(negatives[:, 0], negatives[:, 1], 'ko', markerfacecolor='y', markersize=7, label="Not accepted")
    ax.legend()
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")

    return ax


def plot_decision_boundary(theta, x, y):
    ax = plot_data(x[:, 1:3], y)

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.size, v.size))
    for i in range(u.size):
        for j in range(v.size):
            value = map_feature(u[i], v[j]).dot(theta)[0]
            z[i, j] = value

    z = z.transpose()
    ax.contour(u, v, z, 0, linewidths=2)
    ax.legend()


def map_feature(x1, x2):
    m = x1.size
    degree = 6

    features = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            feature = pow(x1, i - j) * (pow(x2, j))
            features = np.c_[features, feature]

    return features


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def cost_function_reg(theta, x, y, reg_lambda):
    m = y.size
    # ensure it's a vector not an array for dot with reshape below
    h = sigmoid(x.dot(theta.reshape((theta.size, 1))))
    regularization = (reg_lambda / (2 * m)) * (theta[1:] ** 2).sum()

    y_transpose = y.transpose()
    gradients = np.zeros(theta.size)

    j = (-y_transpose.dot(np.log(h)) - ((1 - y_transpose).dot(np.log(1 - h)))) / m
    j = j + regularization

    error_transpose = (h - y).transpose()
    for i in range(theta.size):
        gradient = error_transpose.dot(x[:, i]) / m

        if i > 0:
            gradient_reg = ((reg_lambda / m) * theta[i])
            gradient = gradient + gradient_reg

        gradients[i] = gradient

    return (j, gradients)


def predict(theta, x):
    return sigmoid(x.dot(theta)) > 0.5


plot_data(X, y)
plt.show()

X = map_feature(X[:, 0], X[:, 1])
m, n = X.shape
initial_theta = np.zeros((n, 1))
reg_lambda = 1.0

cost, gradients = cost_function_reg(initial_theta, X, y, reg_lambda)

print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:')
print(' {} '.format(gradients[0:5]))
print('Expected gradients (approx) - first five values only:')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')


# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((n, 1))
cost, gradients = cost_function_reg(test_theta, X, y, 10)

print('Cost at test theta (with lambda = 10): {}'.format(cost))
print('Expected cost (approx): 3.16')
print('Gradient at test theta - first five values only:')
print(' {} '.format(gradients[0:5]))
print('Expected gradients (approx) - first five values only:')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922')


initial_theta = np.zeros((n, 1))
reg_lambda = 1
result = op.minimize(lambda t: cost_function_reg(t, X, y, reg_lambda), initial_theta, method="Newton-CG", jac=True, options={'maxiter': 400, 'disp': True})
cost = result.fun
theta = result.x

plot_decision_boundary(theta, X, y)
plt.show()

# Compute accuracy on our training set
p = predict(theta, X)

accuracy = (p == y[:, 0]).mean() * 100
print('Train Accuracy: {}'.format(accuracy))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')


