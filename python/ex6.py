import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

def read_data(filename):
    data = sio.loadmat('../octave/ex6/{}.mat'.format(filename))
    x = data['X']
    y = data['y']

    return (x, y)

def plot_data(x, y):
    positive = np.nonzero(y == 1)[0]
    negative = np.nonzero(y == 0)[0]

    fig, ax = plt.subplots()
    ax.plot(x[positive, 0], x[positive, 1], 'k+', linewidth=1, markersize=7, label="Positive")
    ax.plot(x[negative, 0], x[negative, 1], 'ko', markerfacecolor='y', markersize=7, label="Negative")

    return ax

def visualize_boundary_linear(x, y, model):
    xp = np.linspace(X.min(), X.max(), 100)
    yp = - ((model.coef_[0][0] * xp) + model.intercept_) / model.coef_[0][1]

    ax = plot_data(x, y)
    ax.plot(xp, yp, '-b')

X, y = read_data('ex6data1')

# plot_data(X, y)
# plt.show()

c = 1
model = svm.LinearSVC(C=c, verbose=True) #, tol=1e-3, max_iter=20)
model.fit(X, y)
print('')

# visualize_boundary_linear(X, y, model)
# plt.show()


def gaussian_kernel(x1, x2, sigma):
    x1 = x1.reshape((-1, 1))
    x2 = x2.reshape((-1, 1))
    sim = np.exp(-((x1 - x2) ** 2).sum() / (2 * (sigma ** 2)))

    return sim


x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {} :'.format(sigma))
print('\t{}'.format(sim))
print('for sigma = 2, this value should be about 0.324652')

X, y = read_data('ex6data2')
# The scipy SVC is much more sensitive to the data scale than the Octave on in the exercises
# However I've commented this out as using a larger sigma / gamma gets the same results as Octave
# X = StandardScaler().fit_transform(X)
# plot_data(X, y)
# plt.show()

c = 1.0
sigma = 0.1
# The sklean SVC doesn't have a sigma parameter, gamma=2.0 gets very close to
# what Octave outputs though when the data is scaled. When it's not scaled below applies.
# gamma is multiplied by the squared euclidian distance, whereas sigma is squared, times, 2, then divided.
# so 0.1 should be a gamma of 50
# When the data is scaled then gamma can be much smaller
gamma = 1 / (2 * (sigma ** 2))
model = svm.SVC(C=c, kernel='rbf', gamma=gamma, verbose=True)
model.fit(X, y)
print('')

def visualize_boundary(x, y, model):
    xxplot = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    yyplot = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
    xx, yy = np.meshgrid(xxplot, yyplot)
    z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    ax = plot_data(x, y)
    ax.contour(xx, yy, z, [0.5], colors='b')

# visualize_boundary(X, y, model)
# plt.show()

X, y = read_data('ex6data3')
# plot_data(X, y)
# plt.show()
X = StandardScaler().fit_transform(X)
y = y.flatten()

# from http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, n_jobs=4)
grid.fit(X, y)
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
