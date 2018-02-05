import tensorflow as tf

import inputs
import os
from datetime import datetime
import pandas as pd
from configuration import ModelConfig
from MLmodel import MLClassifier


FLAGS = tf.app.flags.FLAGS

IMAGE_DIR = "data/VOCdevkit/VOC2012/JPEGImages/"
TFR_DIR = "data/output/"
VAL_FILE_PATTERN = "val-???-001.tfr"
BATCH_SIZE = 1


tf.app.flags.DEFINE_string('model_str', 'evaluate_distort',
                            """model name to store in csv.""")



def main(_):

    config = ModelConfig("inference")
    config.keep_prob = 1.0  # desactivate the dropout

    test_images, test_annotations = inputs.input_pipeline(TFR_DIR, VAL_FILE_PATTERN,
                                    num_classes=config.num_classes, batch_size=BATCH_SIZE)

    data = tf.placeholder(tf.string, [])
    target = tf.placeholder(tf.float32, [None, config.num_classes])

    model = MLClassifier(config, data, target)
    model.build()

    sess = tf.Session()

    sess.run(tf.local_variables_initializer())
    model.restore_fc(sess)
    model.restore_inception(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    num_steps = int((1028)/BATCH_SIZE) # (1 shards * nb examples per shard)

    for n in range(num_steps):
        print("Running inference on image %d" % n)
        image, annotation = sess.run([test_images, test_annotations])
        with open(os.path.join(IMAGE_DIR, image[0].decode('utf8')), 'rb') as f:
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
