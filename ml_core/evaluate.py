import tensorflow as tf

import inputs
from datetime import datetime
from vocabulary import Vocabulary
from CNNS import CNNSigmoid

NUM_STEPS = 10
LOG_DIR = 'models/model'


def main(_):

    if not tf.gfile.IsDirectory(LOG_DIR):
        tf.logging.info("Creating output directory: %s" % LOG_DIR)
        tf.gfile.MakeDirs(LOG_DIR)

    test_images, test_annotations = inputs.input_pipeline('test-???-004.tfr', 32)

    num_classes = len(Vocabulary('mirflickrdata/output/word_counts.txt').vocab)

    data = tf.placeholder(tf.float32, [None, 299, 299, 3])
    target = tf.placeholder(tf.float32, [None, num_classes])

    model = CNNSigmoid(data, target)
    model.build()

    saver = tf.train.Saver()

    # Create the graph, etc.
    init_op = tf.initialize_all_variables()

    # Create a session for running operations in the Graph.
    sess = tf.Session()

    # Initialize the variables (like the epoch counter).
    sess.run(init_op)
    sess.run(tf.initialize_local_variables())
    print("restoring model")
    saver.restore(sess, "models/model/model.ckpt")
    print("Model restored.")

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Run training steps or whatever
    for n in xrange(NUM_STEPS):
        images, annotations = sess.run([test_images, test_annotations])
        sess.run(model.metrics_op, {data: images, target: annotations})
        recall, accuracy, precision = sess.run(model.metrics_value)
        print("%s - Recall : %f Accuracy : %f Precision: %f" % (datetime.now(), recall, accuracy, precision))

    coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
