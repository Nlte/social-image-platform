import tensorflow as tf

import inputs
from datetime import datetime
from image_processing import process_image
from vocabulary import Vocabulary
from MLmodel import MLClassifier
from configuration import ModelConfig
import numpy as np
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image', '',
                            """Image to run inference on.""")

config = ModelConfig("inference")

<<<<<<< HEAD
data = tf.placeholder(tf.string, [])
target = tf.placeholder(tf.float32, [None, config.num_classes])
=======
model = CNNSigmoid(config)
>>>>>>> origin/master

model = MLClassifier(config, data, target)
model.build()

sess = tf.Session()
model.restore_inception(sess)
model.restore_fc(sess)

def inference(filename):
    with open(filename, 'rb') as f:
        encoded_image = f.read()
    dummy_annot = np.zeros((1,len(model.vocabulary.vocab)))
    image, preds = sess.run([model.images, model.prediction],
                            {data: encoded_image, target: dummy_annot})
    idx = [i for i, x in enumerate(preds[0]) if x == 1]
    words = [config.vocabulary.id_to_word(x) for x in idx]
    print(words)

    fig = plt.figure()
    a=fig.add_subplot(1,1,1)
    imgplot = plt.imshow(image[0])
    a.set_title(', '.join(words))
    plt.show()

    return ' '.join(words)


if __name__ == '__main__':
    inference(FLAGS.image)
