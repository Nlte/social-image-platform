import tensorflow as tf

import inputs
from datetime import datetime
from image_processing import process_image
from vocabulary import Vocabulary
from MLmodel import MLClassifier
from configuration import ServerConfig
import numpy as np

class PredictionServer(object):

    def __init__(self):
        self.config = ServerConfig("inference")
        self.data = tf.placeholder(tf.string, [])
        self.target = tf.placeholder(tf.float32, [None, self.config.num_classes])
        model = MLClassifier(self.config, self.data, self.target)
        model.build()
        self.sess = tf.Session()
        model.restore_inception(self.sess)
        model.restore_fc(self.sess)
        self.model = model

    def inference(self, filename):
        with open(filename, 'rb') as f:
            encoded_image = f.read()
        dummy_annot = np.zeros((1, len(self.model.vocabulary.vocab)))
        preds = self.sess.run(self.model.prediction,
                        {self.data: encoded_image, self.target: dummy_annot})
        idx = [i for i, x in enumerate(preds[0]) if x == 1]
        words = [self.config.vocabulary.id_to_word(x) for x in idx]
        return words
