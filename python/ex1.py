import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots()


def plot_data(x, y):
    ax.plot(x, y, 'rx', markersize=10)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    # plt.show()


def plot_linear_fit(x, y):
    ax.plot(x, y, '-')
    # plt.show()


# J = sum(((X * theta) - y). ^ 2) / (2 * m);
def compute_cost(X, y, theta):
    m = y.size
    errors = X.dot(theta) - y
    squared_errors = pow(errors, 2)
    cost = squared_errors.sum() / (2 * m)
    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    cost_history = []
    for iter in range(num_iters):
        h = X.dot(theta)
        errors = h - y
        theta = theta - (alpha * (X.transpose().dot(errors) / m))
        cost_history.append(compute_cost(X, y, theta))
    return (theta, cost_history)


df = pd.read_csv("../octave/ex1/ex1data1.txt", names=['X', 'y'])
X = df['X']
y = df['y']

plot_data(X, y)

m = X.size

df['bias'] = 1
X = df.as_matrix(['bias', 'X'])
y = df.as_matrix(['y'])
theta = np.zeros((2, 1))

J = compute_cost(X, y, theta)
print("With theta = [0; 0] cost computed = {cost}".format(cost=J))
print("Expected cost value (approx) 32.07")

X = df.as_matrix(['bias', 'X'])
y = df.as_matrix(['y'])
theta = np.array([[-1.0], [2.0]])
J = compute_cost(X, y, theta)
print("With theta = [-1 ; 2]\nCost computed = {cost}".format(cost=J))
print("Expected cost value (approx) 54.24")

X = df.as_matrix(['bias', 'X'])
y = df.as_matrix(['y'])
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:')
print('{}'.format(theta))
print('Expected theta values (approx)')
print(' -3.6303  1.1664')

plot_linear_fit(df['X'], df.as_matrix(['bias', 'X']).dot(theta))

# Predict values for population sizes of 35,000 and 70,000
predict1 = (np.matrix([1, 3.5]) * theta)[0, 0]
print('For population = 35,000, we predict a profit of {}'.format(predict1 * 10000))
predict2 = (np.matrix([1, 7]) * theta)[0, 0]
print('For population = 70,000, we predict a profit of {}'.format(predict2 * 10000))

print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_values = np.linspace(-10, 10, 100)
theta1_values = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
j_values = np.zeros((theta0_values.size, theta1_values.size))

# Fill out J_vals
for i in range(theta0_values.size):
    for j in range(theta1_values.size):
        t = np.array([[theta0_values[i]], [theta1_values[j]]])
        j_values[i,j] = compute_cost(X, y, t)

j_values = j_values.transpose()

# TODO: Figure out why this looks flipped compared to Octave
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(theta0_values, theta1_values, j_values, rcount=100, ccount=100)
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.contour(theta0_values, theta1_values, j_values, np.logspace(-2, 3, 20))
ax.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)

plt.show()
