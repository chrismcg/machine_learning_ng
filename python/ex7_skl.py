# Just going to do the image compression in this version
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imageio as iio

image = iio.imread('../octave/ex7/bird_small.png')

image = image / 255 # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size = image.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
x = image.reshape((img_size[0] * img_size[1], 3))

k = 16
clusterer = KMeans(k, init='random', n_jobs=-1)

print("Performing K-Means")
kmeans = clusterer.fit(x)

print("Compressing image")
x_recovered = kmeans.cluster_centers_[kmeans.predict(x)]
x_recovered = x_recovered.reshape((img_size[0], img_size[1], 3))

# Display the original image 
ax = plt.subplot(1, 2, 1)
ax.imshow(image)
ax.set_axis_off()
plt.title('Original')

# Display compressed image side by side
ax = plt.subplot(1, 2, 2)
ax.imshow(x_recovered)
ax.set_axis_off()
plt.title('Compressed, with {} colors.'.format(k))
plt.show()
