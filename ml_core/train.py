import tensorflow as tf

import inputs
import os
from datetime import datetime
from configuration import ModelConfig
from CNNS import CNNSigmoid


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'models/model', """logging directory""")

tf.app.flags.DEFINE_integer('num_steps', 1000, """Number of batches to run.""")

tf.app.flags.DEFINE_string('train_file_pattern', 'train-???-008.tfr',
                            """file pattern of training tfrecords.""")

tf.app.flags.DEFINE_string('val_file_pattern', 'val-???-001.tfr',
                            """file pattern of training tfrecords.""")


def main(_):

    if not tf.gfile.IsDirectory(FLAGS.log_dir):
        tf.logging.info("Creating output directory: %s" % FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

    config = ModelConfig()

    train_images, train_annotations = inputs.input_pipeline(FLAGS.train_file_pattern,
                                    num_classes=config.num_classes, batch_size=16)

    val_images, val_annotations = inputs.input_pipeline(FLAGS.val_file_pattern,
                                    num_classes=config.num_classes, batch_size=16)

    data = tf.placeholder(tf.float32, [None, 299, 299, 3])
    target = tf.placeholder(tf.float32, [None, config.num_classes])

    model = CNNSigmoid("train", config, data, target)
    model.build()

    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    model.init_fn(sess)

    train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("%s Start training." % datetime.now())
    for n in xrange(FLAGS.num_steps):

        if n%50 == 0:
            images, annotations = sess.run([val_images, val_annotations])

            precision, recall, accuracy, f1_score, summary = sess.run(
                [model.precision, model.recall, model.accuracy, model.f1_score, merged],
                {data: images, target: annotations})
            test_writer.add_summary(summary, n)
            print("%s - Precision : %f Recall: %f Accuracy: %f F1score: %f" %
                    (datetime.now(), precision, recall, accuracy, f1_score))

        else:

            images, annotations = sess.run([train_images, train_annotations])
            _, loss, accuracy, precision, recall, f1_score, summary = sess.run(
                [model.optimize, model.loss, model.accuracy,
                model.precision, model.recall, model.f1_score, merged],
                {data:images, target: annotations})

            train_writer.add_summary(summary, n)
            print("%s - Loss : %f Acc: %f" %
                (datetime.now(), loss, accuracy))


    save_path = saver.save(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
