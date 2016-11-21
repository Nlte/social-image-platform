from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import matplotlib

tf.logging.set_verbosity(tf.logging.INFO)


def distort_image(image):
    """Apply random distortions to an image."""
    
    # Randomly flip horizontally.
    with tf.name_scope("flip_horizontal"):
        image = tf.image.random_flip_left_right(image)

    # Randomly distort the colors based on thread idself.
    with tf.name_scope("distortion"):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.032)

        image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def process_image(encoded_image, resize_height=299, resize_width=299, distort=False):
    """Decode and resize image."""

    # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1].
    with tf.name_scope("decode"):
        image = tf.image.decode_jpeg(encoded_image, channels=3)
        else:
            raise ValueError("Invalid image format: %s" % image_format)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Resize image.
    assert (resize_height > 0) == (resize_width > 0)
    image = tf.image.resize_images(image,
                                    size=[resize_height, resize_width],
                                    method=tf.image.ResizeMethod.BILINEAR)

    if distort:
        image = distort_image(image)

    return image
