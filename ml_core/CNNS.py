from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import inputs
import cnn
import image_processing

tf.logging.set_verbosity(tf.logging.INFO)


class CNNSigmoid(object):

    def __init__(self, mode, config, images=None, annotations=None):

        # config
        self.mode = mode
        self.config = config
        # inputs
        self.images = images
        self.annotations = annotations
        # vocabulary
        self.vocabulary = self.config.vocabulary
        # inception
        self.bottleneck_tensor = None
        # sigmoid
        self.loss = None
        self.logits = None
        self.prediction = None
        self.optimize = None
        # evaluation
        self.metrics_op = None
        self.metrics_value = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.accuracy = None


    def build_inputs(self):
        image_feed = tf.placeholder(tf.string, shape=[], name="image_feed")
        annotation_feed = tf.placeholder(tf.float32, shape=[None], name="annotation_feed")
        image = tf.expand_dims(
                image_processing.process_image(image_feed, 299, 299), 0)
        annotation = tf.expand_dims(annotation_feed, 0)
        self.images = image
        self.annotations = annotation


    def build_inception(self):
        cnn.maybe_download_and_extract(self.config.inception_dir)
        inception_output = cnn.inception_v3(self.images, trainable=False)
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.VARIABLES, scope="InceptionV3")
        self.bottleneck_tensor = inception_output


    def build_sigmoid(self):
        output_dim = self.config.num_classes
        bottleneck_dim = self.config.bottleneck_dim
        with tf.name_scope('sigmoid_layer'):
            W = tf.Variable(
                tf.random_normal([bottleneck_dim, output_dim],
                stddev=0.01),
                name='sigmoid_weights'
                )
            b = tf.Variable(tf.zeros([output_dim]), name='sigmoid_biases')
            # logits
            logits = tf.matmul(self.bottleneck_tensor, W) + b
            # compute the activations
            sigmoid_tensor = tf.nn.sigmoid(logits, name='sigmoid_tensor')
            # label is true if sigmoid activation > 0.5
            prediction = tf.round(sigmoid_tensor, name='prediction_tensor')
            cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits, self.annotations)
            )
            tf.scalar_summary('cross_entropy', cross_entropy)
            train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(cross_entropy)
        self.loss = cross_entropy
        self.prediction = prediction
        self.optimize = train_step


    def build_evaluation(self):
        with tf.name_scope('metrics'):
            y_pred = tf.cast(self.prediction, tf.bool)
            not_y_pred = tf.logical_not(y_pred)
            y_true = tf.cast(self.annotations, tf.bool)
            not_y_true = tf.logical_not(y_true)

            tp = tf.reduce_mean(tf.cast(tf.logical_and(y_true, y_pred), tf.float32))
            tn = tf.reduce_mean(tf.cast(tf.logical_and(not_y_pred, y_true), tf.float32))
            fp = tf.reduce_mean(tf.cast(tf.logical_and(not_y_pred, not_y_true), tf.float32))
            fn = tf.reduce_mean(tf.cast(tf.logical_and(y_pred, not_y_true), tf.float32))

            precision = tf.div(tp, tp + fp)
            recall = tf.div(tp, tp + fn)
            f1_score = tf.scalar_mul(2, tf.div(tf.mul(precision, recall), precision + recall))
            accuracy = tf.div(tp + tn, tp + tn + fp + fn)

        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.f1_score = f1_score


    def init_fn(self, sess):
        saver = tf.train.Saver(self.inception_variables)
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
            self.config.inception_checkpoint)
        saver.restore(sess, self.config.inception_checkpoint)


    def setup_global_step(self):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step = global_step


    def build(self):
        """Build the layers of the model."""

        if self.mode == "inference":
            self.build_inputs()

        self.build_inception()
        self.build_sigmoid()
        self.build_evaluation()
        self.setup_global_step()
        tf.logging.info("Model sucessfully built.")


    def restore(self, sess):
        """Restore variables from the checkpoint file in configuration.py"""

        saver = tf.train.Saver()
        tf.logging.info("Restoring model variables from checkpoint file %s",
            self.config.model_checkpoint)
        saver.restore(sess, self.config.model_checkpoint)
        tf.logging.info("Model restored.")
