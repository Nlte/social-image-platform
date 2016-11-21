import tensorflow as tf

import inputs
import os
import shutil
from datetime import datetime
from configuration import ModelConfig
from MLmodel import MLClassifier


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'models/model', """logging directory""")

tf.app.flags.DEFINE_integer('num_epoch', 10, """Number of epoch to run.""")

tf.app.flags.DEFINE_integer('batch_size', 64, """Batch size.""")

tf.app.flags.DEFINE_string('train_file_pattern', 'train-???-008.tfr',
                            """file pattern of training tfrecords.""")

tf.app.flags.DEFINE_string('val_file_pattern', 'val-???-001.tfr',
                            """file pattern of training tfrecords.""")

tf.app.flags.DEFINE_string('bottleneck_dir', 'mirflickrdata/bottlenecks',
                            """bottleneck cache directory.""")

tf.app.flags.DEFINE_string('image_dir', 'mirflickrdata/images',
                            """image directory.""")

tf.app.flags.DEFINE_string('mode', 'train',
                            """Mode of the model : inference, train, benchmark.""")



def main(_):

    if tf.gfile.IsDirectory(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    if not tf.gfile.IsDirectory(FLAGS.log_dir):
        tf.logging.info("Creating output directory: %s" % FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

    config = ModelConfig(FLAGS.mode)

    train_images, train_annotations = inputs.input_pipeline(FLAGS.train_file_pattern,
                                    num_classes=config.num_classes, batch_size=FLAGS.batch_size)

    val_images, val_annotations = inputs.input_pipeline(FLAGS.val_file_pattern,
                                    num_classes=config.num_classes, batch_size=2056)

    data = tf.placeholder(tf.float32, [None, config.bottleneck_dim])
    target = tf.placeholder(tf.float32, [None, config.num_classes])

    model = MLClassifier(config, data, target)
    model.build()

    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("%s Start training." % datetime.now())

    num_batch_per_epoch = int((9 * 2056)/FLAGS.batch_size)  # (nb shards * nb examples per shard) / batch size
    num_steps = FLAGS.num_epoch * num_batch_per_epoch

    for n in xrange(num_steps):
        images, annotations = sess.run([train_images, train_annotations])
        bottlenecks = inputs.get_bottlenecks(images)
        fetches = {'opt':model.optimize, 'loss':model.loss, 'summary': merged}
        feed_dict = {data: bottlenecks, target: annotations}
        v = sess.run(fetches, feed_dict)
        if n%50 == 0:
            train_writer.add_summary(v['summary'], n)

        if n%125 == 0:
            images, annotations = sess.run([val_images, val_annotations])
            bottlenecks = inputs.get_bottlenecks(images)
            fetches = {'auc_ops': model.auc_op, 'summary': merged}
            feed_dict = {data: bottlenecks, target: annotations}
            v = sess.run(fetches, feed_dict)
            mean_auc = sess.run(model.mean_auc)
            print("%s - Validation : Mean AUC : %f" %
                    (datetime.now(), mean_auc))
            test_writer.add_summary(v['summary'], n)

    save_path = saver.save(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
