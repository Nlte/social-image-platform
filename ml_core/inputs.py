from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

tf.logging.set_verbosity(tf.logging.INFO)


def parse_sequence_example(serialized, image_feature, annotation_feature):
    """Parse a sequence example proto : returns the image name and the k-hot annotation."""

    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features={
            image_feature: tf.FixedLenFeature([], dtype=tf.string),

        },
        sequence_features={
            annotation_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })
    image = context[image_feature]
    annotation = sequence[annotation_feature]

    return image, annotation


def read_and_decode(filename_queue):
    """Read filename queue and return the content of sequence examples."""

    reader = tf.TFRecordReader()
    _, record_string = reader.read(filename_queue)
    image, annotation = parse_sequence_example(record_string, 'image/filename', 'image/annotation_bin')
    annotation_float = tf.cast(annotation, tf.float32)

    return image, annotation_float


def input_pipeline(tfr_dir, file_pattern, num_classes, batch_size):
    """Create an input batch queue for one file pattern (ex: val-???-001.tfr)"""

    filenames = []
    filenames.extend(tf.gfile.Glob(os.path.join(tfr_dir, file_pattern)))
    if not filenames:
        tf.logging.fatal("No input files matching %s" % file_pattern)
    else:
        tf.logging.info("Prefetching examples from %s" % file_pattern)
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    image, annotation = read_and_decode(filename_queue)
    annotation.set_shape([num_classes])
    image_batch, annot_batch = tf.train.batch([image, annotation], batch_size=batch_size)

    return image_batch, annot_batch


def get_bottlenecks(images, bottleneck_dir):
    """Returns the bottlenecks for the list of images in argument."""

    bottlenecks = []
    for image in images:
        image = image.decode('utf8')
        bottleneck_path = os.path.join(bottleneck_dir, image+'.txt')
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        bottlenecks.append(bottleneck_values)

    return bottlenecks
