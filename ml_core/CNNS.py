from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import inputs
import cnn
import image_processing
import math

tf.logging.set_verbosity(tf.logging.INFO)


class CNNSigmoid(object):

    def __init__(self, config):

        # config
        self.config = config
        self.mode = self.config.mode
        # inputs
        self.images = None
        self.annotations = None
        # vocabulary
        self.vocabulary = self.config.vocabulary
        # inception
        self.bottleneck_tensor = None
        # sigmoid
        self.keep_prob = self.config.keep_prob
        self.loss = None
        self.logits = None
        self.activations = None
        self.prediction = None
        self.optimize = None
        # evaluation
        self.auc_op_rack = None
        self.auc_rack = None
        self.mean_auc = None
        self.exact_mr = None


    def build_inputs(self):
        if self.mode == "inference":
            image_feed = tf.placeholder(tf.string, shape=[], name="image_feed")
            annotation_feed = tf.placeholder(tf.float32,
                shape=[None, self.config.num_classes], name="annotation_feed")
            image = tf.expand_dims(
                    image_processing.process_image(image_feed, 299, 299), 0)
            self.images = image
            self.annotations = annotation_feed

        elif self.mode in ["train", "test", "benchmark"]:
            bottleneck_feed = tf.placeholder(tf.float32,
                shape=[None, self.config.bottleneck_dim], name="bottleneck_feed")
            annotation_feed = tf.placeholder(tf.float32,
                shape=[None, self.config.num_classes], name="annotation_feed")
            self.bottleneck_tensor = bottleneck_feed
            self.annotations = annotation_feed


    def build_inception(self):
        cnn.maybe_download_and_extract(self.config.inception_dir)
        inception_output = cnn.inception_v3(self.images, trainable=self.config.train_inception)
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.VARIABLES, scope="InceptionV3")
        self.bottleneck_tensor = inception_output


    def build_sigmoid(self):
        output_dim = self.config.num_classes
        bottleneck_dim = self.config.bottleneck_dim

        with tf.variable_scope('sigmoid_layer') as scope:
            W = tf.Variable(
                tf.truncated_normal([bottleneck_dim, output_dim],
                                    stddev=1.0 / math.sqrt(float(bottleneck_dim))),
                name='weights')
            b = tf.Variable(tf.zeros([output_dim]), name=scope.name)
            logits = tf.matmul(self.bottleneck_tensor, W) + b
            sigmoid_tensor = tf.nn.sigmoid(logits, name=scope.name)
            prediction = tf.round(sigmoid_tensor, name=scope.name) # label is true if sigmoid activation > 0.5
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.annotations)
            cross_entropy_sum = tf.reduce_sum(cross_entropy, 1)
            cross_entropy_mean = tf.reduce_mean(cross_entropy_sum)
            tf.scalar_summary('loss', cross_entropy_mean)
            train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(cross_entropy_mean)
        self.fc_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="sigmoid_layer")
        self.loss = cross_entropy_mean
        self.activations = sigmoid_tensor
        self.prediction = prediction
        self.optimize = train_step


    def build_mlp(self):
        """Build the fully connected layers."""

        with tf.name_scope('fully_connected'):

            output_dim = self.config.num_classes
            bottleneck_dim = self.config.bottleneck_dim
            keep_prob = self.keep_prob

            hidden1_units = self.config.hidden1_dim
            hidden2_units = self.config.hidden2_dim

            # Hidden 1
            with tf.variable_scope('hidden1'):
                W = tf.Variable(
                    tf.truncated_normal([bottleneck_dim, hidden1_units],
                                        stddev=1.0 / math.sqrt(float(bottleneck_dim))),
                                        name='weights')
                b = tf.Variable(tf.zeros([hidden1_units]),
                                name='biases')
                hidden1 = tf.nn.relu(tf.matmul(self.bottleneck_tensor, W) + b)
                hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

            with tf.variable_scope('hidden2'):
                W = tf.Variable(
                    tf.truncated_normal([hidden1_units, hidden2_units],
                                        stddev=1.0 / math.sqrt(float(hidden1_units))),
                                        name='weights')
                b = tf.Variable(tf.zeros([hidden2_units]),
                                name='biases')
                hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W) + b)
                hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

            # Linear
            with tf.name_scope('sigmoid'):
                W = tf.Variable(
                    tf.truncated_normal([hidden2_units, output_dim],
                                        stddev=1.0 / math.sqrt(float(hidden2_units))),
                    name='weights')
                b = tf.Variable(tf.zeros([output_dim]),
                                     name='biases')
                logits = tf.matmul(hidden2_drop, W) + b
                sigmoid_tensor = tf.nn.sigmoid(logits, name='sigmoid_activations')

            prediction = tf.round(sigmoid_tensor) # label is true if sigmoid activation > 0.5
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.annotations)
            cross_entropy_sum = tf.reduce_sum(cross_entropy, 1)
            cross_entropy_mean = tf.reduce_mean(cross_entropy_sum)
            tf.scalar_summary('loss', cross_entropy_mean)
            train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(cross_entropy_mean)

        self.fc_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="fully_connected")
        self.loss = cross_entropy_mean
        self.activations = sigmoid_tensor
        self.prediction = prediction
        self.optimize = train_step


    def build_auc(self):
        """Build the tensors computing PR AUC."""

        with tf.name_scope("metrics"):
            correct_prediction = tf.equal(self.prediction, self.annotations)
            all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
            exact_mr = tf.reduce_mean(all_labels_true)
            tf.scalar_summary('exact_mr', exact_mr)
            activation_rack = tf.unpack(tf.cast(self.activations, tf.float32), axis=1)
            label_rack = tf.unpack(tf.cast(self.annotations, tf.float32), axis=1)
            auc_rack = []
            auc_op_rack = []
            i=0
            for activation, label in zip(activation_rack, label_rack):
                auc, auc_op = tf.contrib.metrics.streaming_auc(activation, label, curve='PR')
                auc_rack.append(auc)
                auc_op_rack.append(auc_op)
                i+=1
            mean_auc = tf.reduce_mean(auc_rack)
            tf.scalar_summary('mean_auc', mean_auc)
        self.exact_mr = exact_mr
        self.auc_rack = auc_rack
        self.mean_auc = mean_auc
        self.auc_op_rack = auc_op_rack


    def build(self):
        """Build the layers of the model."""

        if self.mode == "inference":
            self.build_inputs()
            self.build_inception()
            self.build_mlp()
            self.build_auc()

        elif self.mode in ["train", "test"]:
            self.build_inputs()
            self.build_mlp()
            self.build_auc()

        elif self.mode == "benchmark":
            self.build_inputs()
            self.build_sigmoid()
            self.build_auc()

        tf.logging.info("Model sucessfully built.")


    def restore_inception(self, sess):
        """Restore varibales for Inception CNN."""

        saver = tf.train.Saver(self.inception_variables)
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
            self.config.inception_checkpoint)
        saver.restore(sess, self.config.inception_checkpoint)


    def restore_fc(self, sess):
        """Restore variables for the mlp."""

        saver = tf.train.Saver(self.fc_variables)
        tf.logging.info("Restoring model variables from checkpoint file %s",
            self.config.model_checkpoint)
        saver.restore(sess, self.config.model_checkpoint)
        tf.logging.info("Model restored.")
