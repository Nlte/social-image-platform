from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base


DATA_URL = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"


def maybe_download_and_extract(dest_directory):
    """Download and extract Inceptionv3 pretrained checkpoint."""
    
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                        (filename,
                        float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                             filepath,
                                             _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


slim = tf.contrib.slim

def inception_v3(images,
                 trainable,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV3"):
    """Builds an Inception V3 subgraph for image embeddings.

    Args:
        images: A float32 Tensor of shape [batch, height, width, channels].
        trainable: Whether the inception submodel should be trainable or not.
        is_training: Boolean indicating training mode or not.
        weight_decay: Coefficient for weight regularization.
        stddev: The standard deviation of the trunctated normal weight initializer.
        dropout_keep_prob: Dropout keep probability.
        use_batch_norm: Whether to use batch normalization.
        batch_norm_params: Parameters for batch normalization. See
            tf.contrib.layers.batch_norm for details.
        add_summaries: Whether to add activation summaries.
        scope: Optional Variable scope.

        Returns:
            end_points: A dictionary of activations from inception_v3 layers.
    """
    # Only consider the inception model to be in training mode if it's trainable.
    is_inception_model_training = trainable

    if use_batch_norm:
        # Default parameters for batch normalization.
        if not batch_norm_params:
            batch_norm_params = {
                "is_training": is_inception_model_training,
                "trainable": trainable,
                # Decay for the moving averages.
                "decay": 0.9997,
                # Epsilon to prevent 0s in variance.
                "epsilon": 0.001,
                # Collection containing the moving mean and moving variance.
                "variables_collections": {
                    "beta": None,
                    "gamma": None,
                    "moving_mean": ["moving_vars"],
                    "moving_variance": ["moving_vars"],
                }
            }
    else:
        batch_norm_params = None

    if trainable:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None

    with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=weights_regularizer,
            trainable=trainable):
            with slim.arg_scope([slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
                net, end_points = inception_v3_base(images, scope=scope)
                with tf.variable_scope("logits"):
                    shape = net.get_shape()
                    net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                    net = slim.dropout(
                        net,
                        keep_prob=dropout_keep_prob,
                        is_training=is_inception_model_training,
                        scope="dropout")
                    net = slim.flatten(net, scope="flatten")

    return net
