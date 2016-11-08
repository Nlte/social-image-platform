from __future__ import absolute_import, unicode_literals
from celery import Task
from celery import shared_task

import tensorflow as tf

import inputs
from datetime import datetime
from image_processing import process_image
from vocabulary import Vocabulary
from CNNS import CNNSigmoid
from configuration import ModelConfig
import numpy as np
from img_platform.celery import app

class PredictionServer(Task):

    def __init__(self):
        print("\n Building tensorflow model.\n")
        self.config = ModelConfig()
        model = CNNSigmoid("inference", self.config)
        model.build()
        self.sess = tf.Session()
        model.restore(self.sess)
        self.model = model

    def inference(self, filename):
        with open(filename, 'rb') as f:
            encoded_image = f.read()
        dummy_annot = np.zeros(len(self.model.vocabulary.vocab))
        preds = self.sess.run(self.model.prediction,
                        {"image_feed:0": encoded_image, "annotation_feed:0": dummy_annot})
        idx = [i for i, x in enumerate(preds[0]) if x == 1]
        words = [self.config.vocabulary.id_to_word(x) for x in idx]
        return ' '.join(words)
