import tensorflow as tf
import os
import sys
import numpy as np
from fnmatch import fnmatch
from MLmodel import MLClassifier
from configuration import ModelConfig


BOTTLENECK_DIR = "data/bottlenecks/"
IMAGE_DIR = "data/mirflickr/"


def cache_bottlenecks(bottleneck_dir, image_dir):
    """Run each image of image_dir in Inceptionv3 and save the image
        embedding vector in bottleneck_dir."""

    if not tf.gfile.IsDirectory(bottleneck_dir):
        tf.logging.info("Creating output directory: %s" % bottleneck_dir)
        tf.gfile.MakeDirs(bottleneck_dir)

    config = ModelConfig("inference")

    data = tf.placeholder(tf.string, [])
    target = tf.placeholder(tf.float32, [None, config.num_classes])

    model = MLClassifier(config, data, target)
    model.build()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    model.restore_inception(sess)

    # create list of images
    images = []
    for img in os.listdir(image_dir):
        if fnmatch(img, '.*'):
            continue
        if os.path.isdir(os.path.join(image_dir, img)):
            continue
        images.append(img)

    # run inception on the images
    how_many_bottlenecks = 0
    for image in images:
        bottleneck_path = os.path.join(bottleneck_dir, image+'.txt')

        if not os.path.exists(bottleneck_path):
            image_path = os.path.join(image_dir, image)
            with tf.gfile.GFile(image_path, "rb") as f:
                encoded_image = f.read()
            bottleneck_values = sess.run(model.bottleneck_tensor,
                feed_dict={data: encoded_image})
            bottleneck_values = np.squeeze(bottleneck_values)
            bottleneck_string = ','.join(str(x) for x in bottleneck_values)
            bottleneck_path = os.path.join(bottleneck_dir, image+'.txt')
            with open(bottleneck_path, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)
            sys.stdout.write('\r >> Created bottleneck %s' % image+'.txt')
            sys.stdout.flush()

        how_many_bottlenecks += 1
    print(str(how_many_bottlenecks) + ' bottleneck files created.')
    sess.close()


if __name__ == '__main__':
    cache_bottlenecks(BOTTLENECK_DIR, IMAGE_DIR)
