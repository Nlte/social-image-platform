import tensorflow as tf

import inputs
import os
from datetime import datetime
import pandas as pd
from configuration import ModelConfig
from MLmodel import MLClassifier


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('file_pattern', 'val-???-001.tfr',
                            """file pattern of test tfrecords.""")

tf.app.flags.DEFINE_integer('batch_size', 1, """Batch size.""")

tf.app.flags.DEFINE_string('image_dir', 'mirflickrdata/images',
                            """image directory.""")

tf.app.flags.DEFINE_string('model_str', 'evaluate_distort',
                            """model name to store in csv.""")



def main(_):

    config = ModelConfig("inference")
    config.keep_prob = 1.0  # desactivate the dropout

    test_images, test_annotations = inputs.input_pipeline(FLAGS.test_file_pattern,
                                    num_classes=config.num_classes, batch_size=FLAGS.batch_size)

    data = tf.placeholder(tf.string, [])
    target = tf.placeholder(tf.float32, [None, config.num_classes])

    model = MLClassifier(config, data, target)
    model.build()

    sess = tf.Session()

    sess.run(tf.initialize_local_variables())
    model.restore_fc(sess)
    model.restore_inception(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("%s Running test..." % datetime.now())

    num_steps = int((1028)/FLAGS.batch_size) # (1 shards * nb examples per shard)

    for n in xrange(num_steps):
        print("Running inference on image %d" % n)
        image, annotation = sess.run([test_images, test_annotations])
        with open(FLAGS.image_dir+'/'+image[0], 'r') as f:
            image_data = f.read()

        fetches = {'auc_ops': model.auc_op}
        feed_dict = {data: image_data, target: annotation}
        v = sess.run(fetches, feed_dict)

    fetches = {'mean_auc': model.mean_auc, 'auc': model.auc}
    v = sess.run(fetches)
    print("Mean AUC %f" % v['mean_auc'])
    print("Labels AUC")
    print(v['auc'])

    # store result in csv
    dataframe = {'model':FLAGS.model_str,
                'mean AUC': v['mean_auc']}
    for i, k in enumerate(v['auc']):
        label = model.vocabulary.id_to_word(i)
        dataframe[label] = k
    if not os.path.isfile('results.csv'):
        df = pd.DataFrame(dataframe, columns=dataframe.keys(), index=[0])
        df.to_csv('results.csv', index=False)
    else:
        df = pd.read_csv('results.csv')
        len_df = len(df)
        df_buffer = pd.DataFrame(dataframe, columns=dataframe.keys(), index=[len_df+1])
        concat = pd.concat([df, df_buffer])
        concat.to_csv('results.csv', index=False)


    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
