import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tensorflow.examples.tutorials.mnist import input_data

# %%

# Start a tensorflow session
session = tf.InteractiveSession()



# Set the random seed to enable reproducible code
np.random.seed(0)


# %%
mnist = input_data.read_data_sets('MNIST_data')

#%%

img = mnist.test.images[0].reshape(28, 28)
value =  mnist.test.labels[0]
print('center of mass = {}'.format(scipy.ndimage.measurements.center_of_mass(img)))
print('min value = {}, max value = {}'.format(np.min(img), np.max(img)))
plt.imshow(img, cmap='Greys_r');
plt.title('digit = {}'.format(value))
plt.show()