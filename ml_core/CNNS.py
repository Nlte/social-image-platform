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
        self.accuracy = None
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
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.annotations)
            cross_entropy_sum = tf.reduce_sum(cross_entropy, 1)
            cross_entropy_mean = tf.reduce_mean(cross_entropy_sum)
            tf.scalar_summary('loss', cross_entropy_mean)
            train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(cross_entropy_mean)
        self.loss = cross_entropy_mean
        self.prediction = prediction
        self.optimize = train_step


    def build_evaluation(self):
        with tf.name_scope('metrics'):
            y = tf.cast(self.prediction, tf.bool)
            z = tf.cast(self.annotations, tf.bool)

            card_y = tf.reduce_sum(tf.cast(self.prediction, tf.float32), 1)
            card_z = tf.reduce_sum(tf.cast(self.annotations, tf.float32), 1)

            intersection = tf.reduce_sum(tf.to_float(tf.logical_and(y, z)), 1)
            union = tf.reduce_sum(tf.to_float(tf.logical_or(y, z)), 1)

            accuracy = tf.reduce_mean(tf.div(intersection, union))
            precision = tf.reduce_mean(tf.div(intersection, card_z))
            recall = tf.reduce_mean(tf.div(intersection, card_y))
            f1_score = tf.scalar_mul(2, tf.div(tf.mul(precision, recall), precision + recall))

            tf.scalar_summary('precision', precision)
            tf.scalar_summary('recall', recall)
            tf.scalar_summary('accuracy', accuracy)
            tf.scalar_summary('F1-score', f1_score)

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
