import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

data = io.loadmat("../octave/ex3/ex3data1.mat")
X = data['X']
y = data['y']

m, n = X.shape

print("m: {}, n: {}".format(m, n))

random_indices = np.random.permutation(m)
selection = X[random_indices[:100], :]


def display_data(selection):
    example_width = 20
    example_height = 20
    example_shape = (example_width, example_height)

    display_rows = 10
    display_cols = 10

    for i in range(display_rows):
        for j in range(display_cols):
            k = (i * display_rows) + j
            plot = plt.subplot(display_rows, display_cols, k + 1)
            plot.set_axis_off()
            row = selection[k]
            max = row.max()
            image = row.reshape(example_shape, order="F") / max
            plot.imshow(image, cmap="gray", vmin=-1, vmax=1)


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def cost_function_reg(theta, x, y, reg_lambda):
    m = y.size
    # ensure it's a vector not an array for dot with reshape below
    h = sigmoid(x.dot(theta.reshape((theta.size, 1))))

    y_transpose = y.transpose()
    j = (-y_transpose.dot(np.log(h)) - ((1 - y_transpose).dot(np.log(1 - h)))) / m

    regularization = (reg_lambda / (2 * m)) * (theta[1:] ** 2).sum()
    j = j + regularization

    gradients = (x.transpose().dot(h - y)) / m
    temp = theta.copy().reshape((theta.size, 1))
    temp[0] = 0
    gradients = gradients + ((reg_lambda / m) * temp)

    return (j[0][0], gradients.flatten())


def one_vs_all(x, y, num_labels, reg_lambda):
    m, n = x.shape
    all_theta = np.zeros((num_labels, n + 1))

    x = np.c_[np.ones((m, 1)), x]

    for c in range(1, num_labels + 1):
        initial_theta = np.zeros((n + 1, 1))
        # Picked TNC as that gives the closest approximation to the Octave answer
        # Newton-CG got "stuck". L-BFGS-B gave best & fastest answer
        result = op.minimize(
            lambda t: cost_function_reg(t, x, (y == c), reg_lambda),
            initial_theta,
            method="TNC",
            jac=True,
            options={'maxiter': 50, 'disp': True}
        )
        all_theta[c - 1, :] = result.x

    return all_theta


def predict_one_vs_all(all_theta, x):
    m, n = x.shape

    x = np.c_[np.ones((m, 1)), x]

    position = x.dot(all_theta.transpose()).argmax(1)
    return position + 1


display_data(selection)
plt.show()

print('Testing cost_function_reg')

theta_t = np.array([-2, -1, 1, 2]).reshape((4, 1))
X_t = np.c_[np.ones((5, 1)), np.array(range(1, 16)).reshape((5, 3), order='F') / 10]
y_t = np.array([[1], [0], [1], [0], [1]])
lambda_t = 3
j, gradients = cost_function_reg(theta_t, X_t, y_t, lambda_t)

cost = j

print('Cost: {}'.format(cost))
print('Expected cost: 2.534819')
print('Gradients:')
print(' {} '.format(gradients))
print('Expected gradients:')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003')


num_labels = 10
reg_lambda = 0.1
all_theta = one_vs_all(X, y, num_labels, reg_lambda)

predictions = predict_one_vs_all(all_theta, X)

accuracy = (predictions == y[:, 0]).mean() * 100
print('Training Set Accuracy: {}'.format(accuracy))
