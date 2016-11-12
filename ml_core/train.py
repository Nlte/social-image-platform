import tensorflow as tf

import inputs
import os
from datetime import datetime
from configuration import ModelConfig
from CNNS import CNNSigmoid


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'models/model', """logging directory""")

tf.app.flags.DEFINE_integer('num_epoch', 10, """Number of epoch to run.""")

tf.app.flags.DEFINE_integer('batch_size', 32, """Batch size.""")

tf.app.flags.DEFINE_string('train_file_pattern', 'train-???-008.tfr',
                            """file pattern of training tfrecords.""")

tf.app.flags.DEFINE_string('val_file_pattern', 'val-???-001.tfr',
                            """file pattern of training tfrecords.""")

tf.app.flags.DEFINE_string('bottleneck_dir', 'mirflickrdata/bottlenecks',
                            """bottleneck cache directory.""")

tf.app.flags.DEFINE_string('image_dir', 'mirflickrdata/images',
                            """image directory.""")



def main(_):

    if not tf.gfile.IsDirectory(FLAGS.log_dir):
        tf.logging.info("Creating output directory: %s" % FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)


    config = ModelConfig()

    train_images, train_annotations = inputs.input_pipeline(FLAGS.train_file_pattern,
                                    num_classes=config.num_classes, batch_size=FLAGS.batch_size)

    val_images, val_annotations = inputs.input_pipeline(FLAGS.val_file_pattern,
                                    num_classes=config.num_classes, batch_size=FLAGS.batch_size)

    #data = tf.placeholder(tf.float32, [None, 299, 299, 3])
    #target = tf.placeholder(tf.float32, [None, config.num_classes])

    model = CNNSigmoid("train", config)
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

        if n%500 == 0:
            images, annotations = sess.run([val_images, val_annotations])
            bottlenecks = inputs.get_bottlenecks(images)
            fetches = {'auc_ops': model.auc_op_rack, 'mean_auc': model.mean_auc, 'summary': merged}
            feed_dict = {'bottleneck_feed:0': bottlenecks, 'annotation_feed:0': annotations}
            v = sess.run(fetches, feed_dict)
            print("%s - Validation : Mean AUC %f" % (datetime.now(), v['mean_auc']))
            test_writer.add_summary(v['summary'], n)


        images, annotations = sess.run([train_images, train_annotations])
        bottlenecks = inputs.get_bottlenecks(images)
        fetches = {'opt':model.optimize, 'loss':model.loss, 'mean_auc':model.mean_auc,
                    'auc_ops':model.auc_op_rack, 'summary': merged}
        feed_dict = {'bottleneck_feed:0': bottlenecks, 'annotation_feed:0': annotations}
        v = sess.run(fetches, feed_dict)
        if n%100 == 0:
            print("%s - Step %d - Loss : %f mean AUC: %f" %
                (datetime.now(), n, v['loss'], v['mean_auc']))
            train_writer.add_summary(v['summary'], n)

    save_path = saver.save(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
