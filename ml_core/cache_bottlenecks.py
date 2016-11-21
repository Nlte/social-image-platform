import tensorflow as tf
import os
import sys
import numpy as np
from fnmatch import fnmatch
from MLmodel import MLClassifier
from configuration import ModelConfig


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('bottleneck_dir', 'mirflickrdata/bottlenecks-noncentered',
                            """bottleneck cache directory.""")

tf.app.flags.DEFINE_string('image_dir', 'mirflickrdata/images',
                            """image directory.""")


def cache_bottlenecks(bottleneck_dir, image_dir):

    if not tf.gfile.IsDirectory(bottleneck_dir):
        tf.logging.info("Creating output directory: %s" % bottleneck_dir)
        tf.gfile.MakeDirs(bottleneck_dir)

    config = ModelConfig("inference")

    model = MLClassifier(config)
    model.build()

    sess = tf.Session()

    sess.run(tf.initialize_all_variables())
    model.restore_inception(sess)

    # create list of images
    images = []
    for img in os.listdir(image_dir):
        if fnmatch(img, '.*'):
            continue
        images.append(img)

    # run inception on the images
    how_many_bottlenecks = 0
    for image in images:
        bottleneck_path = os.path.join(bottleneck_dir, image+'.txt')

        if not os.path.exists(bottleneck_path):
            image_path = os.path.join(image_dir, image)
            with tf.gfile.GFile(image_path, "r") as f:
                encoded_image = f.read()
            bottleneck_values = sess.run(model.bottleneck_tensor,
                feed_dict={"image_feed:0": encoded_image})
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
    cache_bottlenecks(FLAGS.bottleneck_dir, FLAGS.image_dir)

from fnmatch import fnmatch
