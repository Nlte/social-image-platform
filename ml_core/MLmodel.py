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


def linear(x, name, size):
    """Returns linear layer."""

    W = tf.get_variable(name+"/W", [x.get_shape()[1], size])
    b = tf.get_variable(name+"/b", [size], initializer=tf.zeros_initializer)
    tf.histogram_summary(name+"/W", W)
    tf.histogram_summary(name+"/b", b)

    return tf.matmul(x, W) + b


class MLClassifier(object):
    """Multilabel Classifier object."""

    def __init__(self, config, data, target):

        # config
        self.config = config
        self.mode = self.config.mode
        self.vocabulary = self.config.vocabulary
        # inputs
        if self.mode == "inference":
            self.images = data

        if self.mode == "train":
            self.bottleneck_tensor = data

        self.annotations = target
        # FC
        self.loss = None
        self.logits = None
        self.activations = None
        self.prediction = None
        self.optimize = None
        # evaluation
        self.auc_op = None
        self.auc = None
        self.mean_auc = None


    def build_preprocess(self):
        """Build the image preprocessing ops."""

        with tf.name_scope("preprocessing"):
            images = tf.expand_dims(
                    image_processing.process_image(self.images,distort=self.config.distort_image),0)
            self.images = images


    def build_inception(self):
        """Build the inception v3 CNN."""

        cnn.maybe_download_and_extract(self.config.inception_dir)
        inception_output = cnn.inception_v3(self.images, trainable=self.config.train_inception)
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.VARIABLES, scope="InceptionV3")
        self.bottleneck_tensor = inception_output


    def build_fc(self):
        """Build the fully connected layers."""

        with tf.name_scope('fully_connected') as scope:

            output_dim = self.config.num_classes
            bottleneck_dim = self.config.bottleneck_dim
            hidden_dim = self.config.hidden_dim
            keep_prob = self.config.keep_prob

            net = self.bottleneck_tensor
            # hidden
            for i in range(self.config.num_hidden):
                net = tf.nn.relu(
                    linear(net, scope+"linear_%d" % i, hidden_dim)
                    )
                if keep_prob < 1.0:
                    net = tf.nn.dropout(net, keep_prob)
            # linear
            net = linear(net, scope+"regression", self.config.num_classes)
            self.activations = tf.nn.sigmoid(net)
            self.prediction = tf.round(self.activations)

            cross_entropy = tf.reduce_sum(
                            tf.nn.sigmoid_cross_entropy_with_logits(net, self.annotations), 1)
            self.loss = tf.reduce_mean(cross_entropy)
            tf.scalar_summary("loss", self.loss)
            self.optimize = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

            self.fc_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="fully_connected")


    def build_metrics(self):
        """Build the tensors computing PR AUC."""

        with tf.name_scope("metrics"):
            activation_rack = tf.unpack(tf.cast(self.activations, tf.float32), axis=1)
            label_rack = tf.unpack(tf.cast(self.annotations, tf.float32), axis=1)
            auc = []
            auc_op = []
            i=0
            for activation, label in zip(activation_rack, label_rack):
                value, op = tf.contrib.metrics.streaming_auc(activation, label, curve='PR')
                auc_op.append(op)
                auc.append(value)
                i+=1

            mean_auc = tf.reduce_mean(auc)
            tf.scalar_summary("mean_auc", mean_auc)

        self.mean_auc = mean_auc
        self.auc_op = auc_op
        self.auc = auc



    def build(self):
        """Build the layers of the model."""

        if self.mode == "inference":
            self.build_preprocess()
            self.build_inception()
            self.build_fc()
            self.build_metrics()

        elif self.mode == "train":
            self.build_fc()
            self.build_metrics()

        tf.logging.info("Model sucessfully built.")


    def restore_inception(self, sess):
        """Restore Inception variables from checkpoint."""

        saver = tf.train.Saver(self.inception_variables)
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
            self.config.inception_checkpoint)
        saver.restore(sess, self.config.inception_checkpoint)
        tf.logging.info("Inception-v3 restored")


    def restore_fc(self, sess):
        """Restore fully connected variables from checkpoint."""

        saver = tf.train.Saver(self.fc_variables)
        tf.logging.info("Restoring model variables from checkpoint file %s",
            self.config.model_checkpoint)
        saver.restore(sess, self.config.model_checkpoint)
        tf.logging.info("FC restored")
