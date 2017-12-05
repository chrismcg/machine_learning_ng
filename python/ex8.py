import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from math import pi

#% ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easy to
#  visualize.
#
#  Our example case consists of 2 network server statistics across
#  several machines: the latency and throughput of each machine.
#  This exercise will help us find possibly faulty (or very fast) machines.
#

print('Visualizing example dataset for outlier detection.')

#  The following command loads the dataset. You should now have the
#  variables X, Xval, yval in your environment
data = sio.loadmat('../octave/ex8/ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bx')
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

#% ================== Part 2: Estimate the dataset statistics ===================
#  For this exercise, we assume a Gaussian distribution for the dataset.
#
#  We first estimate the parameters of our assumed Gaussian distribution, 
#  then compute the probabilities for each of the points and then visualize 
#  both the overall distribution and where each of the points falls in 
#  terms of that distribution.
#
print('Visualizing Gaussian fit.')

#  Estimate my and sigma2
mu = np.mean(X, axis=0)
sigma2 = np.var(X, axis=0)

def multivariate_gaussian(x, mu, sigma2):
    k = mu.size
    x = x - mu
    sigma2 = np.diag(sigma2)
    p = (2 * pi) ** (-k / 2) * np.linalg.det(sigma2) ** -0.5 * np.exp(-0.5 * (x.dot(np.linalg.pinv(sigma2)) * x).sum(axis=1))

    return p

#  Returns the density of the multivariate normal at each data point (row) 
#  of X
p = multivariate_gaussian(X, mu, sigma2)

def visualize_fit(x, mu, sigma):
    grid_coords = np.arange(0, 35.5, 0.5)
    xx, yy = np.meshgrid(grid_coords, grid_coords)
    grid_params = np.c_[xx.ravel(), yy.ravel()]
    z = multivariate_gaussian(grid_params, mu, sigma)
    z = z.reshape(xx.shape)

    plt.plot(x[:, 0], x[:, 1], 'bx')
    if np.isinf(z).sum() == 0:
        contours = 10.0 ** np.arange(-20, 0, 3)
        plt.contour(xx, yy, z, contours)

#  Visualize the fit
visualize_fit(X,  mu, sigma2);
plt.xlabel('Latency (ms)');
plt.ylabel('Throughput (mb/s)');
plt.show()

#% ================== Part 3: Find Outliers ===================
#  Now you will find a good epsilon threshold using a cross-validation set
#  probabilities given the estimated Gaussian distribution
# 

def select_threshold(yval, pval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    step_size = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step_size):
        prediction = pval < epsilon

        true_positives = np.logical_and((prediction == 1), (yval == 1)).sum()
        false_positives = np.logical_and((prediction == 1), (yval == 0)).sum()
        true_negatives = np.logical_and((prediction == 0), (yval == 0)).sum()
        false_negatives = np.logical_and((prediction == 0), (yval == 1)).sum()

        if (true_positives + false_positives == 0):
            continue

        if (true_positives + false_negatives == 0):
            continue

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = (2.0 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return (best_epsilon, best_f1)

pval = multivariate_gaussian(Xval, mu, sigma2)

epsilon, F1 = select_threshold(yval.T, pval)
print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set:  {}'.format(F1))
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)')

#  Find the outliers in the training set and plot the
outliers = np.nonzero(p < epsilon)

#  Draw a red circle around those outliers
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=10)
visualize_fit(X,  mu, sigma2);
plt.xlabel('Latency (ms)');
plt.ylabel('Throughput (mb/s)');
plt.show()

#% ================== Part 4: Multidimensional Outliers ===================
#  We will now use the code from the previous part and apply it to a 
#  harder problem in which more features describe each datapoint and only 
#  some features indicate whether a point is an outlier.
#

#  Loads the second dataset. You should now have the
#  variables X, Xval, yval in your environment
data = sio.loadmat('../octave/ex8/ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

#  Apply the same steps to the larger dataset
mu = np.mean(X, axis=0)
sigma2 = np.var(X, axis=0)

#  Training set 
p = multivariate_gaussian(X, mu, sigma2)

#  Cross-validation set
pval = multivariate_gaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = select_threshold(yval.T, pval)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set:  %f' % F1)
print('   (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of 0.615385)')
print('# Outliers found: %d' % p[np.nonzero(p < epsilon)].sum())
