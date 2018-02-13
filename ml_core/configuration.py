from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from vocabulary import Vocabulary
import os.path as path


class ModelConfig(object):
    """Wrapper class for the configuration of training models."""

    def __init__(self, mode):

        ## directories and files ##
        # working directory
        dirname = path.dirname(__file__)
        # directory to store the inception v3 checkpoint
        self.inception_dir = path.join(dirname,"models/inception")
        # inception v3 checkpoint file
        self.inception_checkpoint = path.join(dirname,"models/inception/inception_v3.ckpt")
        # mlp checkpoint file
        self.model_checkpoint = path.join(dirname,"models/model/model.ckpt")
        # list of labels
        self.vocab_file = path.join(dirname,"data/word_counts.txt")

        ## model ##
        self.mode = mode
        assert self.mode in ["train", "inference"]
        # Inception fine-tuning
        self.train_inception = False
        # Distort input image
        self.distort_image = False
        # Learning rate of the optimizer
        self.learning_rate = 5e-4
        # Dimension of the image embedding vector
        self.bottleneck_dim = 2048
        # Number of hidden layers
        self.num_hidden = 1
        # Dimension of hidden layers
        self.hidden_dim = 1500
        # Kepp prob for the dropout layer
        self.keep_prob = 0.9
        # create the Vocabulary object
        self.vocabulary = Vocabulary(self.vocab_file)
        # output dimension (total number of labels in the dataset)
        self.num_classes = len(self.vocabulary.vocab)


class ServerConfig(object):
    """Wrapper class for the configuration of production model. (running on the website)"""

    def __init__(self, mode):

        ## directories and files ##

        dirname = path.dirname(__file__)

        self.inception_dir = path.join(dirname,"models/inception")

        self.inception_checkpoint = path.join(dirname,"models/inception/inception_v3.ckpt")

        self.model_checkpoint = path.join(dirname,"models/prod_model/model.ckpt")

        self.vocab_file = path.join(dirname,"models/prod_model/word_counts.txt")

        ## model ##
        self.mode = "inference"

        self.train_inception = False

        self.distort_image = False

        self.learning_rate = 5e-4

        self.bottleneck_dim = 2048

        self.num_hidden = 1

        self.hidden_dim = 1500

        self.keep_prob = 0.9

        self.vocabulary = Vocabulary(self.vocab_file)

        self.num_classes = len(self.vocabulary.vocab)
