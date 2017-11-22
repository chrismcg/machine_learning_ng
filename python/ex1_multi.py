import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    df = pd.read_csv("../octave/ex1/ex1data2.txt", names=['X1', 'X2', 'y'])
    x = df.as_matrix(['X1', 'X2'])
    y = df.as_matrix(['y'])

    return (x, y)


def feature_normalize(x):
    mu = x.mean(0)
    sigma = x.std(0)
    x_norm = (x - mu) / sigma

    return (x_norm, mu, sigma)


def compute_cost_multi(x, y, theta):
    m = y.size
    errors = X.dot(theta) - y
    squared_errors = pow(errors, 2)
    cost = squared_errors.sum() / (2 * m)
    return cost


def gradient_descent_multi(x, y, theta, alpha, iterations):
    new_theta = theta
    j_history = []
    for i in range(iterations):
        h = x.dot(new_theta)
        errors = h - y
        new_theta = new_theta - (alpha * (x.transpose().dot(errors) / m))

        j_history.append(compute_cost_multi(x, y, new_theta))

    return (new_theta, j_history)


def normal_equation(x, y):
    x_prime = x.transpose()
    theta = np.linalg.pinv(x_prime.dot(x)).dot(x_prime).dot(y)

    return theta


X, y = read_data()
m = y.size

print('First 10 examples from the dataset: ')
for i in range(10):
    print(" x = [{} {}], y = {}".format(X[i][0], X[i][1], y[i]))

normalized_X, mu, sigma = feature_normalize(X)

X = np.c_[np.ones(m), normalized_X]

alpha = 0.01
iterations = 400

theta = np.zeros((3, 1))

theta, j_history = gradient_descent_multi(X, y, theta, alpha, iterations)

plt.plot(list(range(len(j_history))), j_history, '-b', linewidth=2)
plt.show()

print('Theta computed from gradient descent: ')
print(' {}'.format(theta))

to_estimate = np.array([1650, 3])
print(to_estimate)
to_estimate_normalized = (to_estimate - mu) / sigma
print(to_estimate_normalized)
input = np.r_[np.ones(1), to_estimate_normalized]
price = input.dot(theta)[0]

print('Predicted price of a 1650 sq-ft, 3 br house ')
print('(using gradient descent):')
print(' {}'.format(price))

X, y = read_data()
m = y.size
X = np.c_[np.ones(m), X]

theta = normal_equation(X, y)

print('Theta computed from normal equations: ')
print(' {}'.format(theta))

to_estimate = np.array([1, 1650, 3])
price = to_estimate.dot(theta)[0]

print('Predicted price of a 1650 sq-ft, 3 br house ')
print('(using normal equations):')
print(' {}'.format(price))
