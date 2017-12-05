import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

#% ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('Visualizing example dataset for PCA.')

#  The following command loads the dataset. You should now have the 
#  variable X in your environment
data = sio.loadmat('../octave/ex7/ex7data1.mat')
X = data['X']

fig, ax = plt.subplots()

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
plt.show()

#% =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('Running PCA on example dataset.')


def feature_normalize(x):
    mu = x.mean(0)
    x_norm = x - mu
    sigma = x_norm.std(0)
    x_norm = x_norm / sigma

    return (x_norm, mu, sigma)

def pca(x):
    m = x.shape[0]
    sigma = (1 / m) * x.T.dot(x)
    return np.linalg.svd(sigma)

data = sio.loadmat('../octave/ex7/ex7data1.mat')
X = data['X']

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = feature_normalize(X)

#  Run PCA
u, s, _ = pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
def draw_line(ax, p1, p2, *args, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], *args, **kwargs)


fig, ax = plt.subplots()
plt.plot(X[:, 0], X[:, 1], 'bo')
draw_line(ax, mu, mu + 1.5 * s[0] * u[:, 0], '-k', linewidth=2)
draw_line(ax, mu, mu + 1.5 * s[1] * u[:, 1], '-k', linewidth=2)
plt.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
plt.show()

print('Top eigenvector: ');
print(' U[:,0] = {} {}'.format(u[0,0], u[1,0]))
print('you should expect to see -0.707107 -0.707107)')

#% =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('Dimension reduction on example dataset.')

#  Plot the normalized dataset (returned from pca)
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis([-4, 3, -4, 3])
ax.set_aspect('equal')


def project_data(x, u, k):
    u_reduce = u[:, :k]
    z = x.dot(u_reduce)

    return z

def recover_data(z, u, k):
    u_reduce = u[:, :k]
    x_rec = z.dot(u_reduce.T)
    return x_rec

#  Project the data onto K = 1 dimension
k = 1;
z = project_data(X_norm, u, k);
print('Projection of the first example: {}\n'.format(z[0]))
print('(this value should be about 1.481274)')
print('(this is off by ~0.01 in python which I think is numerical)')

X_rec  = recover_data(z, u, k)
print('Approximation of the first example: {} {}'.format(X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)')
print('(this is off by ~0.01 in python which I think is numerical)')

# #  Draw lines connecting the projected points to the original points
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
ax.set_aspect('equal')
for i in range(X_norm.shape[0]):
    draw_line(plt, X_norm[i, :], X_rec[i, :], '--k', linewidth=1)

plt.show()

#% =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('Loading face dataset.')

#  Load Face dataset
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
fig, ax = plt.subplots()
display_data(X[:100, :])
plt.show()

#% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('Running PCA on face dataset.')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, mu, sigma = feature_normalize(X)

#  Run PCA
u, s, _ = pca(X_norm)
#  Visualize the top 36 eigenvectors found
display_data(u[:, :36].T, rows=6, cols=6)
plt.show()


#% ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('Dimension reduction for face dataset.')

k = 100;
z = project_data(X_norm, u, k);

print('The projected data Z has a size of: ')
print('{} '.format(z.shape))

#% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.')

k = 100;
X_rec  = recover_data(z, u, k)

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
