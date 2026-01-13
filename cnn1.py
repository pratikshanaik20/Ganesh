# import the necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# set the param
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

# define the kernel (edge detection)
kernel = tf.constant([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], dtype=tf.float32)

# load the image
image = tf.io.read_file('Ganesha.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[300, 300])

# plot the image
img = tf.squeeze(image).numpy()
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Gray Scale image')
plt.show()

# Reformat
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)  # shape: [1, H, W, 1]

# reshape kernel: [filter_height, filter_width, in_channels, out_channels]
kernel = tf.reshape(kernel, [3, 3, 1, 1])

# convolution layer
conv_fn = tf.nn.conv2d
image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=[1, 1, 1, 1],  # must be 4D
    padding='SAME',
)

plt.figure(figsize=(15, 5))

# Plot the convolved image
plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(image_filter).numpy(), cmap='gray')
plt.axis('off')
plt.title('Convolution')

# activation layer
relu_fn = tf.nn.relu
image_detect = relu_fn(image_filter)

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(image_detect).numpy(), cmap='gray')
plt.axis('off')
plt.title('Activation')

# pooling layer (using tf.nn.pool)
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME',
)

plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(image_condense).numpy(), cmap='gray')
plt.axis('off')
plt.title('Pooling')

plt.show()