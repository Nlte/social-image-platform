from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter
from collections import namedtuple
from datetime import datetime
from nltk.corpus import words
import json
import os
import random
import sys
import re


tf.flags.DEFINE_string('output_dir', 'output',
			"Output directory.")

tf.flags.DEFINE_string('image_dir', 'images',
			"Location of the mirflickr image directory.")

tf.flags.DEFINE_string('annot_file', 'annotations.json',
			"json file containing annotations")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",["filename", "annotation"])

class Vocabulary(object):

    def __init__(self, vocab):
        self._vocab = vocab

    def word_to_id(self, word):
      """Returns the integer id of a word string."""
      if word in self._vocab:
          return self._vocab[word]


def _is_png(filename):
    return '.png' in filename


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _load_and_process_metadata(annotations_file):
    with tf.gfile.FastGFile(annotations_file, "r") as f:
        json_data = json.load(f)
    print("Processing json annotation file.")
    image_metadata = []
    for data in json_data:
        filename = data['image']
        annotation = data['annotation']
        image_metadata.append(ImageMetadata(filename, annotation))
    print("Built %d image metadata objects" % len(json_data))

    return image_metadata


def _create_vocab(annotations):
    print("Creating vocabulary.")
    counter = Counter(annotations)
    print("Total words:", len(counter))

    word_counts = [x for x in counter.items()]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    output_file = FLAGS.output_dir + '/word_counts.txt'
    with tf.gfile.FastGFile(output_file, "w") as f:
      f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file: %s" % output_file)

    reverse_vocab = [x[0] for x in word_counts]
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict)

    return vocab


def get_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def _to_sequence_example(image, vocab):

    image_path = os.path.join(FLAGS.image_dir, image.filename)
    with tf.gfile.FastGFile(image_path, 'r') as f:
        encoded_image = f.read()

    context = tf.train.Features(feature={
        "image/filename": _bytes_feature(image.filename),
    })

    annotation_ids = [vocab.word_to_id(word) for word in image.annotation]
    annotation_bin = [0] * len(vocab._vocab)
    for a in annotation_ids:
        annotation_bin[a] = 1

    feature_lists = tf.train.FeatureLists(feature_list={
        "image/annotation": _bytes_feature_list(image.annotation),
        "image/annotation_ids": _int64_feature_list(annotation_ids),
        "image/annotation_bin": _int64_feature_list(annotation_bin),
    })

    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _process_dataset(name, dataset, vocab, chunk_size):
    random.seed(14352)
    random.shuffle(dataset)
    rng = len(dataset)/chunk_size
    i = 0
    chunks = get_chunks(dataset, chunk_size)
    print("%s Processing dataset." % datetime.now())
    for chunk in chunks:
        output_filename = '%s-%.3d-%.3d.tfr' % (name, i, rng)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        for image in chunk:
            sequence_example = _to_sequence_example(image, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())

        print("%s Processed %d of %d" % (datetime.now(), i, rng))
        i += 1
        writer.close()


def main(_):
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    dataset = _load_and_process_metadata(FLAGS.annotation_file)

    n0 = int(0.8 * len(dataset))
    train_dataset = dataset[:n0]
    test_dataset = dataset[n0:]
    n1 = int(0.90 * len(train_dataset))
    val_dataset = train_dataset[n1:]
    train_dataset = train_dataset[:n1]

    annotations = [x for image in dataset for x in image.annotation]
    vocabulary = _create_vocab(annotations)

    _process_dataset('train', train_dataset, vocabulary, 2056)
    _process_dataset('test', test_dataset, vocabulary, 1028)
    _process_dataset('val', val_dataset, vocabulary, 1028)


if __name__ == "__main__":
    tf.app.run()
