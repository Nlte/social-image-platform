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

OUTPUT_DIR = 'output'
ABS_IMAGE_DIR = os.path.join(os.getcwd(),'images')
CAPTION_FILE = 'annotations.json'
SHARDS_TRAIN = 256
SHARDS_TEST = 8
VAL_SHARDS = 4

ImageMetadata = namedtuple("ImageMetadata",["filename", "annotation", "tags"])

class Vocabulary(object):

    def __init__(self, vocab):
        self._vocab = vocab

    def word_to_id(self, word):
      """Returns the integer id of a word string."""
      if word in self._vocab:
          return self._vocab[word]


class Tags(object):

    def __init__(self, tags, unk_id):
        self._tags = tags
        self._unk_id = unk_id
    def word_to_id(self, tag):
        if tag in self._tags:
            return self._tags[tag]
        else:
            return self._unk_id


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
    print("Processing json annotations file.")
    image_metadata = []
    for data in json_data:
        filename = data['image']
        annotation = data['annotation']
        tags = data['tags']
        tags = [tag.encode('utf-8') for tag in tags]
        image_metadata.append(ImageMetadata(filename, annotation, tags))
    print("Built %d image metadata objects" % len(json_data))

    return image_metadata


def _create_vocab(annotations):
    print("Creating vocabulary.")
    counter = Counter(annotations)
    print("Total words:", len(counter))

    word_counts = [x for x in counter.items()]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    output_file = OUTPUT_DIR + '/word_counts.txt'
    with tf.gfile.FastGFile(output_file, "w") as f:
      f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file: %s" % output_file)

    reverse_vocab = [x[0] for x in word_counts]
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict)

    return vocab


def _create_tags(tags):
    print("Creating tag vocabulary.")
    counter = Counter(tags)
    print("Total words:", len(counter))
    all_words = set(words.words())
    clean_tags = [x for x in counter.items() if x[0] in all_words]
    clean_tags = [x for x in clean_tags if not re.search(r'\d', x[0])]
    word_counts = [x for x in counter.items() if x[1] >= 200]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print(word_counts)
    print("Words in tag vocabulary:", len(word_counts))
    raise SystemExit
    output_file = OUTPUT_DIR + '/tag_counts.txt'
    with tf.gfile.FastGFile(output_file, "w") as f:
      f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote tag file: %s" % output_file)

    reverse_tags = [x[0] for x in word_counts]
    tags_dict = dict([(x, y) for (y, x) in enumerate(reverse_tags)])
    unk_id = len(reverse_tags)
    tags = Tags(tags_dict, unk_id)

    return tags


def get_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def _to_sequence_example(image, vocab, tags):

    image_path = os.path.join(ABS_IMAGE_DIR, image.filename)
    with tf.gfile.FastGFile(image_path, 'r') as f:
        encoded_image = f.read()

    context = tf.train.Features(feature={
        "image/filename": _bytes_feature(image.filename),
        "image/data": _bytes_feature(encoded_image),
    })

    annotation_ids = [vocab.word_to_id(word) for word in image.annotation]
    annotation_bin = [0] * len(vocab._vocab)
    for a in annotation_ids:
        annotation_bin[a] = 1
    tags_ids = [tags.word_to_id(word) for word in image.tags]
    feature_lists = tf.train.FeatureLists(feature_list={
        "image/annotation": _bytes_feature_list(image.annotation),
        "image/annotation_ids": _int64_feature_list(annotation_ids),
        "image/annotation_bin": _int64_feature_list(annotation_bin),
        "image/tags": _bytes_feature_list(image.tags),
        "image/tags_ids": _int64_feature_list(tags_ids)
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _process_dataset(name, dataset, vocab, tags, chunk_size):
    random.seed(14352)
    random.shuffle(dataset)
    rng = len(dataset)/chunk_size
    i = 0
    chunks = get_chunks(dataset, chunk_size)
    print("%s Processing dataset." % datetime.now())
    for chunk in chunks:
        output_filename = '%s-%.3d-%.3d.tfr' % (name, i, rng)
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        for image in chunk:
            sequence_example = _to_sequence_example(image, vocab, tags)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())

        print("%s Processed %d of %d" % (datetime.now(), i, rng))
        i += 1
        writer.close()


def main(_):
    if not tf.gfile.IsDirectory(OUTPUT_DIR):
        tf.gfile.MakeDirs(OUTPUT_DIR)

    dataset = _load_and_process_metadata(CAPTION_FILE)

    n0 = int(0.8 * len(dataset))
    train_dataset = dataset[:n0]
    test_dataset = dataset[n0:]
    n1 = int(0.90 * len(train_dataset))
    val_dataset = train_dataset[n1:]
    train_dataset = train_dataset[:n1]

    annotations = [x for image in dataset for x in image.annotation]
    alltags = [x for image in dataset for x in image.tags]
    vocabulary = _create_vocab(annotations)
    tags = _create_tags(alltags)
    raise SystemExit


    _process_dataset('train', train_dataset, vocabulary, tags, 2056)
    _process_dataset('test', test_dataset, vocabulary, tags, 1028)
    _process_dataset('val', val_dataset, vocabulary, tags, 1028)


if __name__ == "__main__":
    tf.app.run()
