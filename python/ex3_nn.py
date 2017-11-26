import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat("../octave/ex3/ex3data1.mat")
X = data['X']
y = data['y']

input_layer_size = 400
hidden_layer_size = 20
num_labels = 10

weights = sio.loadmat("../octave/ex3/ex3weights.mat")
theta1 = weights['Theta1']
theta2 = weights['Theta2']


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def predict(theta1, theta2, x):
    m, n = x.shape

    a1 = np.c_[np.ones((m, 1)), x]
    z2 = a1.dot(theta1.transpose())

    a2 = sigmoid(z2)
    a2_m, a2_n = a2.shape
    a2 = np.c_[np.ones((a2_m, 1)), a2]

    z3 = a2.dot(theta2.transpose())
    a3 = sigmoid(z3)
    predictions = a3.argmax(1) + 1

    return predictions

def display_prediction(x, y, prediction):
    fig, ax = plt.subplots()
    max = x.max()
    image = x.reshape((20, 20), order="F")
    ax.imshow(image, cmap="gray", vmin=-1, vmax=1)
    ax.set_axis_off()
    plt.title("Actual: {}, Predicted: {}".format(y, prediction))
    plt.show()

predictions = predict(theta1, theta2, X)
accuracy = (predictions == y[:, 0]).mean() * 100
print('Training Set Accuracy: {}'.format(accuracy))

m, n = X.shape
randperm = np.random.permutation(m)

for i in range(m):
    x_t = X[randperm[i]].reshape((1, input_layer_size))
    y_t = y[randperm[i]][0]
    prediction = predict(theta1, theta2, x_t)[0]

    if y_t == 10:
        y_t = 0

    if prediction == 10:
        prediction = 0

    display_prediction(x_t, y_t, prediction)