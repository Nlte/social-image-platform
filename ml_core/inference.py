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
model.restore_inception(sess)
#sess.run(tf.initialize_all_variables())
model.restore_fc(sess)

def inference(filename):
    with open(filename, 'rb') as f:
        encoded_image = f.read()
    dummy_annot = np.zeros((1,len(model.vocabulary.vocab)))
    preds = sess.run(model.prediction,
                    {"image_feed:0": encoded_image, "annotation_feed:0": dummy_annot})
    idx = [i for i, x in enumerate(preds[0]) if x == 1]
    words = [config.vocabulary.id_to_word(x) for x in idx]
    return ' '.join(words)

print(inference('landscapetree.jpg'))
