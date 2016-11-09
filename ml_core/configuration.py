from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from vocabulary import Vocabulary
import os.path as path


class ModelConfig(object):
    """Wrapper class for configuration."""

    def __init__(self):

        # directories and files
        dirname = path.dirname(__file__)

        self.inception_dir = path.join(dirname,"models/inception")

        self.inception_checkpoint = path.join(dirname,"models/inception/inception_v3.ckpt")

        self.model_checkpoint = path.join(dirname,"models/model/model.ckpt")

        self.vocab_file = path.join(dirname,"mirflickrdata/output/word_counts.txt")


        # model
        self.train_inception = True

        self.learning_rate = 0.05

        self.bottleneck_dim = 2048

        self.vocabulary = Vocabulary(self.vocab_file)

        self.num_classes = len(self.vocabulary.vocab)
