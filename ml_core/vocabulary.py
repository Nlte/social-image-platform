from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


class Vocabulary(object):
    """Vocabulary class for an image-to-text model."""
    def __init__(self, vocab_file):
        """Initializes the vocabulary.
        Args:
            vocab_file: File containing the vocabulary, where the words are the first
            whitespace-separated token on each line (other tokens are ignored) and
            the word ids are the corresponding line numbers.
        """
        if not tf.gfile.Exists(vocab_file):
            tf.logging.fatal("Vocab file %s not found.", vocab_file)
        tf.logging.info("Initializing vocabulary from file: %s", vocab_file)

        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
        reverse_vocab = [line.split()[0] for line in reverse_vocab]

        vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

        tf.logging.info("Created vocabulary with %d words" % len(vocab))

        self.vocab = vocab  # vocab[word] = id
        self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word

    def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        return self.vocab[word]

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        return self.reverse_vocab[word_id]
