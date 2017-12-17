import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

data = sio.loadmat('../octave/ex8/ex8data1.mat')
X = data['X']
print(X.shape)
Xval = data['Xval']
yval = data['yval']

# #  Visualize the example dataset
# plt.plot(X[:, 0], X[:, 1], 'bx')
# plt.axis([0, 30, 0, 30])
# plt.xlabel('Latency (ms)')
# plt.ylabel('Throughput (mb/s)')
# plt.show()
# contamination = (yval == 1).sum() / yval.shape[0]
# print("contamination: {}".format(contamination))
# There are 6 outliers
contamination = 6 / X.shape[0]

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('ee', EllipticEnvelope(assume_centered=True, contamination=contamination))
])

print("fitting")
pipeline.fit(X)

def visualize_fit(x, pipeline):
    grid_coords = np.arange(0, 35.5, 0.5)
    xx, yy = np.meshgrid(grid_coords, grid_coords)
    grid_params = np.c_[xx.ravel(), yy.ravel()]
    z = pipeline.decision_function(grid_params)
    z = z.reshape(xx.shape)

    plt.plot(x[:, 0], x[:, 1], 'bx')
    if np.isinf(z).sum() == 0:
        plt.contour(xx, yy, z, levels=[0])

#  Visualize the fit
visualize_fit(X, pipeline)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title("EllipticEnvelope: contamination: {}".format(contamination))

outliers = X[pipeline.predict(X) == -1]
plt.plot(outliers[:, 0], outliers[:, 1], 'ro', linewidth=2, markersize=10)
plt.show()

predicted = pipeline.predict(Xval)
predicted = (predicted == -1).astype(int)
score = f1_score(yval, predicted)
print("f1 score {}:".format(score))

# The way SKL EllipticEnvelope works is different that how the
# multivariate_gaussian works in the original octave. There's no epsilon but
# instead you tell it what percentage of the points are outliers and it works
# out a threshold from that.

# Going to try some of the other approaches SKL has for outlier detection
# rather than re-do that.

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('ocsvm', OneClassSVM(nu=contamination))
])

pipeline.fit(X)
#  Visualize the fit
visualize_fit(X, pipeline)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title("OneClassSVM: nu: {}".format(contamination))

outliers = X[pipeline.predict(X) == -1]
plt.plot(outliers[:, 0], outliers[:, 1], 'ro', linewidth=2, markersize=10)
plt.show()

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('isof', IsolationForest(contamination=contamination))
])

pipeline.fit(X)
#  Visualize the fit
visualize_fit(X, pipeline)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title("IsolationForest: contamination: {}".format(contamination))

outliers = X[pipeline.predict(X) == -1]
plt.plot(outliers[:, 0], outliers[:, 1], 'ro', linewidth=2, markersize=10)
plt.show()
