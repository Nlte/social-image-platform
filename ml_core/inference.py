import tensorflow as tf

import inputs
from datetime import datetime
from image_processing import process_image
from vocabulary import Vocabulary
from CNNS import CNNSigmoid
from configuration import ModelConfig
import numpy as np

print("\n Building tensorflow model.\n")


config = ModelConfig()

model = CNNSigmoid("inference", config)

model.build()

sess = tf.Session()

model.restore(sess)

def inference(filename):
    with open(filename, 'rb') as f:
        encoded_image = f.read()
    image_data = sess.run(processed_image, {image_feed: encoded_image})
    preds = sess.run(model.prediction, {data: image_data, target: dummy_annot})
    idx = [i for i, x in enumerate(preds[0]) if x == 1]
    words = [config.vocabulary.id_to_word(x) for x in idx]
    return words
