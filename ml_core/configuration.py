from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from vocabulary import Vocabulary
import os.path as path


class ModelConfig(object):
    """Wrapper class for configuration."""

    def __init__(self, mode):

        # directories and files
        dirname = path.dirname(__file__)

        self.inception_dir = path.join(dirname,"models/inception")

        self.inception_checkpoint = path.join(dirname,"models/inception/inception_v3.ckpt")

        self.model_checkpoint = path.join(dirname,"models/model/model.ckpt")

        self.vocab_file = path.join(dirname,"mirflickrdata/output/word_counts.txt")

        self.tfr_dir = path.join(dirname,"mirflickrdata/output")

        self.bottleneck_dir = path.join(dirname,"mirflickrdata/bottlenecks-noncentered")


        # model
        self.mode = mode
        assert self.mode in ["train", "inference"]

        self.train_inception = False

        self.distort_image = False

        self.learning_rate = 5e-4

        self.bottleneck_dim = 2048

        self.num_hidden = 1

        self.hidden_dim = 1500

        self.keep_prob = 0.9

        self.vocabulary = Vocabulary(self.vocab_file)

        self.num_classes = len(self.vocabulary.vocab)
