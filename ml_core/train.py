import tensorflow as tf

import inputs
import os
import shutil
from datetime import datetime
from configuration import ModelConfig
from MLmodel import MLClassifier

LOG_DIR = "models/model"
TFR_DIR = "data/output"
BOTTLENECK_DIR = "data/bottlenecks"
TRAIN_FILE_PATTERN = "train-???-008.tfr"
VAL_FILE_PATTERN = "val-???-001.tfr"

NUM_EPOCH = 10
BATCH_SIZE = 64


def main(_):

    if tf.gfile.IsDirectory(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)

    if not tf.gfile.IsDirectory(LOG_DIR):
        tf.logging.info("Creating output directory: %s" % LOG_DIR)
        tf.gfile.MakeDirs(LOG_DIR)

    config = ModelConfig("train")

    train_images, train_annotations = inputs.input_pipeline(TFR_DIR, TRAIN_FILE_PATTERN,
                                    num_classes=config.num_classes, batch_size=BATCH_SIZE)

    val_images, val_annotations = inputs.input_pipeline(TFR_DIR, VAL_FILE_PATTERN,
                                    num_classes=config.num_classes, batch_size=2056)

    data = tf.placeholder(tf.float32, [None, config.bottleneck_dim])
    target = tf.placeholder(tf.float32, [None, config.num_classes])

    model = MLClassifier(config, data, target)
    model.build()

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("%s Start training." % datetime.now())

    num_batch_per_epoch = int((9 * 2056)/BATCH_SIZE)  # (nb shards * nb examples per shard) / batch size
    num_steps = NUM_EPOCH * num_batch_per_epoch

    for n in range(num_steps):
        images, annotations = sess.run([train_images, train_annotations])
        bottlenecks = inputs.get_bottlenecks(images, BOTTLENECK_DIR)
        fetches = {'opt':model.optimize, 'loss':model.loss, 'summary': merged}
        feed_dict = {data: bottlenecks, target: annotations}
        v = sess.run(fetches, feed_dict)
        if n%10 == 0:
            print('Loss: %f' % v['loss'])
            train_writer.add_summary(v['summary'], n)

        if n%100 == 0:
            images, annotations = sess.run([val_images, val_annotations])
            bottlenecks = inputs.get_bottlenecks(images, BOTTLENECK_DIR)
            fetches = {'auc_ops': model.auc_op, 'summary': merged}
            feed_dict = {data: bottlenecks, target: annotations}
            v = sess.run(fetches, feed_dict)
            mean_auc = sess.run(model.mean_auc)
            print("%s - Validation : Mean AUC : %f" %
                    (datetime.now(), mean_auc))
            test_writer.add_summary(v['summary'], n)

    save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
