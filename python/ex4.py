import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

data = sio.loadmat("../octave/ex4/ex4data1.mat")
X = data['X']
y = data['y']
m, n = X.shape

weights = sio.loadmat("../octave/ex4/ex4weights.mat")
theta1 = weights['Theta1']
theta2 = weights['Theta2']
nn_params = np.hstack((theta1.flatten(order='F'), theta2.flatten(order='F')))

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

random_indices = np.random.permutation(m)
image_selection = X[random_indices[:100], :]


def display_data(selection, width=20, height=20, rows=10, cols=10):
    shape = (width, height)

    for i in range(rows):
        for j in range(cols):
            k = (i * rows) + j
            plot = plt.subplot(rows, cols, k + 1)
            plot.set_axis_off()
            row = selection[k]
            max = row.max()
            image = row.reshape(shape, order="F") / max
            plot.imshow(image, cmap="gray", vmin=-1, vmax=1)


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoid_gradient(z):
    value = sigmoid(z)
    return value * (1 - value)


def randomize_initial_weight(layer_in, layer_out):
    epsilon_init = 0.12
    return np.random.rand(layer_out, layer_in + 1) * (2 * epsilon_init) - epsilon_init


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, reg_lambda):
    theta1_end = hidden_layer_size * (input_layer_size + 1)
    theta1 = nn_params[:theta1_end].reshape((hidden_layer_size, input_layer_size + 1), order='F')
    theta2_start = hidden_layer_size * (input_layer_size + 1)
    theta2 = nn_params[theta2_start:].reshape((num_labels, hidden_layer_size + 1), order='F')

    m, n = x.shape
    j = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    y_vectors = (list(range(1, num_labels + 1)) == y) * 1.0

    # forward prop

    a1 = np.c_[np.ones((m, 1)), x]
    z2 = a1.dot(theta1.transpose())

    a2 = sigmoid(z2)

    a2_size, _ = a2.shape
    a2 = np.c_[np.ones((a2_size, 1)), a2]

    z3 = a2.dot(theta2.transpose())
    h_theta = sigmoid(z3)

    positives = -y_vectors * np.log(h_theta)
    negatives = (1 - y_vectors) * np.log(1 - h_theta)
    j = ((positives - negatives) / m).sum()
    theta1_reg = (theta1[:, 1:] ** 2).sum()
    theta2_reg = (theta2[:, 1:] ** 2).sum()
    regularization = (theta1_reg + theta2_reg) * reg_lambda / (2 * m)
    j += regularization

    # back prop
    delta3 = h_theta - y_vectors
    delta2 = (delta3.dot(theta2)) * a2 * (1 - a2)
    delta2 = delta2[:, 1:]

    theta2_grad = theta2_grad + (delta3.transpose().dot(a2))
    theta1_grad = theta1_grad + (delta2.transpose().dot(a1))

    theta2[:, 0] = 0
    theta2_grad = (theta2_grad + (reg_lambda * theta2)) / m
    theta1[:, 0] = 0
    theta1_grad = (theta1_grad + (reg_lambda * theta1)) / m

    unrolled_thetas = np.hstack((theta1_grad.flatten(order='F'), theta2_grad.flatten(order='F')))
    return (j, unrolled_thetas)


def debug_initial_weights(fan_out, fan_in):
    weights = np.zeros((fan_out, fan_in + 1))
    sins = np.sin(range(1, weights.size + 1)).reshape(weights.shape, order="F") / 10
    return sins


def compute_numerical_gradients(cost_function, theta):
    numerical_gradients = np.zeros(theta.size)
    pertubations = np.zeros(theta.size)
    e = 1e-4

    for p in range(theta.size):
        pertubations[p] = e
        loss1, _ = cost_function(theta - pertubations)
        loss2, _ = cost_function(theta + pertubations)
        numerical_gradients[p] = (loss2 - loss1) / (2 * e)
        pertubations[p] = 0

    return numerical_gradients


def check_nn_gradients(reg_lambda=0.0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = debug_initial_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_initial_weights(num_labels, hidden_layer_size)
    X = debug_initial_weights(m, input_layer_size - 1)
    y = 1 + np.mod(range(1, m + 1), num_labels).reshape((m, 1))

    nn_params = np.hstack((theta1.flatten(order='F'), theta2.flatten(order='F')))

    def test_cost_function(p): return nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)
    cost, gradients = test_cost_function(np.copy(nn_params))

    numerical_gradients = compute_numerical_gradients(test_cost_function, np.copy(nn_params))
    print("{}".format(np.c_[numerical_gradients, gradients]))

    diff = np.linalg.norm(numerical_gradients - gradients, ord=2) / np.linalg.norm(numerical_gradients + gradients, ord=2)

    print("If your backpropagation implementation is correct, then\nthe relative difference will be small (less than 1e-9).")
    print("Relative Difference: {}".format(diff))


def predict(theta1, theta2, x):
    m, _ = x.shape
    num_labels, _ = theta2.shape

    h1 = sigmoid(np.c_[np.ones((m, 1)), x].dot(theta1.transpose()))
    h2 = sigmoid(np.c_[np.ones((m, 1)), h1].dot(theta2.transpose()))

    position = h2.argmax(1) + 1
    return position

display_data(image_selection)
plt.show()

reg_lambda = 0
j, _ = nn_cost_function(nn_params.copy(), input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)

print('Cost at parameters (loaded from ex4weights): {}'.format(j))
print('(this value should be about 0.287629)')

reg_lambda = 1
j, _ = nn_cost_function(nn_params.copy(), input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)

print('Cost at parameters (loaded from ex4weights): {}'.format(j))
print('(this value should be about 0.383770)')

print('Evaluating sigmoid gradient...')

g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:  ')
print('{}'.format(g))
print()
print()

print('Initializing Neural Network Parameters ...')

initial_theta1 = randomize_initial_weight(input_layer_size, hidden_layer_size)
initial_theta2 = randomize_initial_weight(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params= np.hstack((initial_theta1.flatten(order='F'), initial_theta2.flatten(order='F')))

print('Checking Backpropagation... ');

#  Check gradients by running checkNNGradients
check_nn_gradients()

print('Checking Backpropagation (w/ Regularization) ...')

#  Check gradients by running checkNNGradients
reg_lambda = 3.0
check_nn_gradients(reg_lambda)

# Also output the costFunction debugging values
debug_J, _ = nn_cost_function(nn_params.copy(), input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)

print('Cost at (fixed) debugging parameters (w/ lambda = {}): {}'.format(reg_lambda, debug_J))
print('(for lambda = 3, this value should be about 0.576051)')


print('Training Neural Network...')
reg_lambda = 1.0

def cost_function(p):
    return nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)


# Newton-CG didn't get "stuck" and gave an answer slightly better than Octave version (though this seems dependent on initial weights)
# L-BFGS-B also worked and was slightly better / faster
result = op.minimize(
            cost_function,
            np.copy(initial_nn_params),
            method="Newton-CG",
            jac=True,
            options={'maxiter': 50, 'disp': True}
        )

cost = result.fun
nn_params = result.x

theta1_end = hidden_layer_size * (input_layer_size + 1)
theta1 = nn_params[:theta1_end].reshape((hidden_layer_size, input_layer_size + 1), order='F')
theta2_start = hidden_layer_size * (input_layer_size + 1)
theta2 = nn_params[theta2_start:].reshape((num_labels, hidden_layer_size + 1), order='F')

display_data(theta1[:, 1:], width=20, height=20, rows=5, cols=5)
plt.show()

predictions = predict(theta1, theta2, X)

accuracy = (predictions == y[:, 0]).mean() * 100
print('Training Set Accuracy: {}'.format(accuracy))
