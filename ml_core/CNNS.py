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
        self.activations = None
        self.prediction = None
        self.optimize = None
        # evaluation
        self.auc_op_rack = None
        self.auc_rack = None
        self.mean_auc = None


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
        with tf.variable_scope('sigmoid_layer') as scope:
            W = tf.Variable(
                tf.random_normal([bottleneck_dim, output_dim],
                stddev=0.01),
                name=scope.name
                )
            b = tf.Variable(tf.zeros([output_dim]), name=scope.name)
            # logits
            logits = tf.matmul(self.bottleneck_tensor, W) + b
            # compute the activations
            sigmoid_tensor = tf.nn.sigmoid(logits, name=scope.name)
            # label is true if sigmoid activation > 0.5
            prediction = tf.round(sigmoid_tensor, name=scope.name)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.annotations)
            cross_entropy_sum = tf.reduce_sum(cross_entropy, 1)
            cross_entropy_mean = tf.reduce_mean(cross_entropy_sum)
            tf.scalar_summary('loss', cross_entropy_mean)
            train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(cross_entropy_mean)
        self.loss = cross_entropy_mean
        self.activations = sigmoid_tensor
        self.prediction = prediction
        self.optimize = train_step


    def build_auc(self):
        activation_rack = tf.unpack(tf.cast(self.activations, tf.float32), axis=1)
        label_rack = tf.unpack(tf.cast(self.annotations, tf.float32), axis=1)
        auc_rack = []
        auc_op_rack = []
        with tf.name_scope('metrics') as scope:
            i=0
            for activation, label in zip(activation_rack, label_rack):
                auc, auc_op = tf.contrib.metrics.streaming_auc(activation, label, curve='PR')
                auc_rack.append(auc)
                auc_op_rack.append(auc_op)
                tf.scalar_summary('auc/'+self.vocabulary.id_to_word(i), auc)
                i+=1
        mean_auc = tf.reduce_mean(auc_rack)
        tf.scalar_summary('mean_auc', mean_auc)
        self.auc_rack = auc_rack
        self.mean_auc = mean_auc
        self.auc_op_rack = auc_op_rack


    def build_summaries(self):
        train_summary_ops = []
        train_summary_ops.append(tf.scalar_summary('loss', self.loss))
        train_summary_ops.append(tf.scalar_summary('mean_auc', self.mean_auc))

        test_summary_ops = []
        i=0
        for auc in self.auc_rack:
            op = tf.scalar_summary('auc/'+self.vocabulary.id_to_word(i), auc)
            test_summary_ops.append(op)
            i+=1
        self.train_summary = train_summary_ops
        self.test_summary = test_summary_ops


    def build(self):
        """Build the layers of the model."""

        if self.mode == "inference":
            self.build_inputs()

        self.build_inception()
        self.build_sigmoid()
        self.build_auc()
        tf.logging.info("Model sucessfully built.")


    def init_fn(self, sess):
        saver = tf.train.Saver(self.inception_variables)
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
            self.config.inception_checkpoint)
        saver.restore(sess, self.config.inception_checkpoint)


    def restore(self, sess):
        """Restore variables from the checkpoint file in configuration.py"""

        saver = tf.train.Saver()
        tf.logging.info("Restoring model variables from checkpoint file %s",
            self.config.model_checkpoint)
        saver.restore(sess, self.config.model_checkpoint)
        tf.logging.info("Model restored.")
