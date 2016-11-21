from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from configuration import ModelConfig
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

config = ModelConfig("train")

TFR_DIR = config.tfr_dir
BOTTLENECK_DIR = config.bottleneck_dir
VOCAB_FILE = config.vocab_file

tf.logging.set_verbosity(tf.logging.INFO)



def parse_sequence_example(serialized, image_feature, annotation_feature):
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
    reader = tf.TFRecordReader()
    _, record_string = reader.read(filename_queue)
    image, annotation = parse_sequence_example(record_string, 'image/filename', 'image/annotation_bin')
    annotation_float = tf.cast(annotation, tf.float32)

    return image, annotation_float


def input_pipeline(file_pattern, num_classes, batch_size):
    filenames = []
    filenames.extend(tf.gfile.Glob(TFR_DIR + '/' + file_pattern))
    if not filenames:
        tf.logging.fatal("No input files matching %s" % file_pattern)
    else:
        tf.logging.info("Prefetching examples from %s" % file_pattern)
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    image, annotation = read_and_decode(filename_queue)
    annotation.set_shape([num_classes])
    image_batch, annot_batch = tf.train.batch([image, annotation], batch_size=batch_size)

    return image_batch, annot_batch


def get_bottlenecks(images):
    bottlenecks = []
    for image in images:
        bottleneck_path = os.path.join(BOTTLENECK_DIR, image+'.txt')
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        bottlenecks.append(bottleneck_values)

    return bottlenecks
