import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#% ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('Visualizing example dataset for PCA.')

#  The following command loads the dataset. You should now have the 
#  variable X in your environment
data = sio.loadmat('../octave/ex7/ex7data1.mat')
X = data['X']


def draw_line(ax, p1, p2, *args, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], *args, **kwargs)


#  Visualize the example dataset
fig, ax = plt.subplots()
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
plt.show()

#% =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('Running PCA on example dataset.')

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
mu = scaler.mean_

pca = PCA()
pca.fit(X_norm)
u = pca.components_
s = pca.explained_variance_

fig, ax = plt.subplots()
plt.plot(X[:, 0], X[:, 1], 'bo')
draw_line(ax, mu, mu + 1.5 * s[0] * u[:, 0], '-k', linewidth=2)
draw_line(ax, mu, mu + 1.5 * s[1] * u[:, 1], '-k', linewidth=2)
plt.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
plt.show()

#% =================== Part 3: Dimension Reduction ===================

pca = PCA(n_components=1)
pca.fit(X_norm)
z = pca.transform(X_norm)
print('Projection of the first example: {}\n'.format(z[0]))
print('(this value should be about 1.481274)')
print('(this is off by ~0.01 in python which I think is numerical)')

X_rec = pca.inverse_transform(z)
print('Approximation of the first example: {} {}'.format(X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)')
print('(this is off by ~0.01 in python which I think is numerical)')

#  Draw lines connecting the projected points to the original points
fig, ax = plt.subplots()
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis([-4, 3, -4, 3])
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
for i in range(X.shape[0]):
    draw_line(plt, X_norm[i, :], X_rec[i, :], '--k', linewidth=1)
ax.set_aspect('equal')

plt.show()

#% =============== Part 4: Loading and Visualizing Face Data =============
data = sio.loadmat('../octave/ex7/ex7faces.mat')
X = data['X']

def display_data(selection, width=32, height=32, rows=10, cols=10):
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

#  Display the first 100 faces in the dataset
# fig, ax = plt.subplots()
# display_data(X[:100, :])
# plt.show()

#% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
print('Running PCA on face dataset.')
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
mu = scaler.mean_

k = 100;
pca = PCA(svd_solver='full', n_components=k)
pca.fit(X_norm)
u = pca.components_
s = pca.explained_variance_
#  Visualize the top 36 eigenvectors found
# display_data(u[:, :36].T, rows=6, cols=6)
# plt.show()

#% ============= Part 6: Dimension Reduction for Faces =================
print('Dimension reduction for face dataset.')

z = pca.transform(X_norm)
X_rec = pca.inverse_transform(z)

fig, ax = plt.subplots()
# Display normalized data
ax = plt.subplot(1, 2, 1);
display_data(X_norm[:100,:])
plt.title('Original faces')
ax.set_aspect('equal')

fig, ax = plt.subplots()
# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
display_data(X_rec[:100,:])
plt.title('Recovered faces');
ax.set_aspect('equal')

plt.show()
