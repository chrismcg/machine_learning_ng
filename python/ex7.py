import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio as iio

#% ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm 
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you should complete the code in the findClosestCentroids function. 
#
print('Finding closest centroids.')

# Load an example dataset that we will be using
data = sio.loadmat('../octave/ex7/ex7data2.mat')
X = data['X']

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]]);

def find_closest_centroids(x, centroids):
    k = centroids.shape[0]
    m = x.shape[0]
    distance_to_centroids = np.zeros((m, k))

    for j in range(k):
        distance_to_centroids[:, j] = np.sqrt(((x - centroids[j, :]) ** 2).sum(axis=1))

    new_centroids = np.argmin(distance_to_centroids, axis=1)

    return new_centroids

# Find the closest centroids for the examples using the
# initial_centroids
idx = find_closest_centroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: ')
print(' {}'.format(idx[:3]))
print('(the closest centroids should be 0, 2, 1 respectively)')

#% ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
print('Computing centroids means.')

def compute_centroids(x, idx, k):
    n = x.shape[1]
    centroids = np.zeros((k, n))
    for i in range(k):
        centroid_points = x[np.nonzero(idx == i), :]
        centroids[i, :] = centroid_points.mean(axis=1)

    return centroids

#  Compute means based on the closest centroids found in the previous part.
centroids = compute_centroids(X, idx, K);

print('Centroids computed after initial finding of closest centroids: ')
print(' {} '.format(centroids))
print('(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]')

#% =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided. 
#
print('Running K-Means clustering on example dataset.')

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]]);

def plot_data_points(ax, x, idx, k):
    normalizer = Normalize(vmin=0, vmax=k-1)
    colors = normalizer(idx)
    ax.scatter(X[:, 0], X[:, 1], s=15, c=colors)


def draw_line(ax, p1, p2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]])


def plot_progress_k_means(ax, x, centroids, previous, idx, k, i):
    plot_data_points(ax, x, idx, k)
    ax.plot(centroids[:, 0], centroids[:, 1], 'x', markeredgecolor='k', markersize=10, linewidth=3)
    for j in range(centroids.shape[0]):
        draw_line(ax, centroids[j, :], previous[j, :])

    plt.title("Iteration number {}".format(i))


def run_k_means(x, initial_centroids, max_iters, plot_progress=False):
    m = x.shape[0]
    k = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))

    if plot_progress:
        plt.ion()
        fig, ax = plt.subplots()
        plt.show()

    for i in range(max_iters):
        print("K-Means iterations {}/{}".format(i, max_iters))
        idx = find_closest_centroids(x, centroids)
        if plot_progress:
            plot_progress_k_means(ax, x, centroids, previous_centroids, idx, k, i)
            plt.draw()
            input("press return to continue")
            previous_centroids = centroids
        centroids = compute_centroids(x, idx, k)

    if plot_progress:
        plt.ioff()
        plt.show() # this waits for user to close figure before continuing

    return (centroids, idx)

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = run_k_means(X, initial_centroids, max_iters, plot_progress=True);
print('K-Means Done.')

#% ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel onto its closest centroid.
#  
#  You should now complete the code in kMeansInitCentroids.m
#

print('Running K-Means clustering on pixels from an image.')

#  Load an image of a bird
a = iio.imread('../octave/ex7/bird_small.png')

a = a / 255 # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size = a.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
x = a.reshape((img_size[0] * img_size[1], 3))

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
k = 16
max_iters = 10


def k_means_init_centroids(x, k):
    rand_index = np.random.permutation(x.shape[0])
    return x[rand_index[:k], :]


# When using K-Means, it is important the initialize the centroids
# randomly. 
# You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = k_means_init_centroids(x, k);

# Run K-Means
centroids, idx = run_k_means(x, initial_centroids, max_iters);

#% ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we 

print('Applying K-Means to compress an image.')

# Find closest cluster members
idx = find_closest_centroids(x, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx. 

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
x_recovered = centroids[idx, :]

# Reshape the recovered image into proper dimensions
x_recovered = x_recovered.reshape((img_size[0], img_size[1], 3))

# Display the original image 
ax = plt.subplot(1, 2, 1)
ax.imshow(a)
ax.set_axis_off()
plt.title('Original')

# Display compressed image side by side
ax = plt.subplot(1, 2, 2)
ax.imshow(x_recovered)
ax.set_axis_off()
plt.title('Compressed, with {} colors.'.format(k))
plt.show(block=True)
