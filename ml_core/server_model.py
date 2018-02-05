import tensorflow as tf
import numpy as np
import inputs
import pdb

from datetime import datetime
from image_processing import process_image
from vocabulary import Vocabulary
from MLmodel import MLClassifier
from configuration import ServerConfig


class ServerModel(object):

    def __init__(self):
        self.config = ServerConfig('inference')
        self.data = tf.placeholder(tf.string, [])
        self.target = tf.placeholder(tf.float32, [None, self.config.num_classes])
        model = MLClassifier(self.config, self.data, self.target)
        model.build()
        self.session = tf.Session()
        model.restore_inception(self.session)
        model.restore_fc(self.session)
        self.model = model

    def inference(self, encoded_image):
        dummy = np.zeros((1, len(self.model.vocabulary.vocab)))
        preds = self.session.run(
            self.model.prediction,
            {self.data: encoded_image, self.target: dummy}
        )
        idx = [i for i,x in enumerate(preds[0]) if x == 1]
        labels = [self.config.vocabulary.id_to_word(x) for x in idx]
        return labels
