# _*_ coding: utf-8 _*_
# @Time : 2020/5/19 上午10:24 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : tf_test.py
import tensorflow as tf

from utils import reader_tfrecord

input_file = '../data/test.tf_record'

coord = tf.train.Coordinator()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
threads = tf.train.start_queue_runners(sess, coord)

train_data = reader_tfrecord(input_file, 8
                             , capacity=64, min_after_dequeue=10)

input_ids = train_data['input_ids']
labels = train_data['labels']
batch_inputs, batch_labels = sess.run([input_ids, labels])
print(batch_inputs)
print(batch_labels)

# 请求线程结束
coord.request_stop()
# 等待线程终止
coord.join(threads)
sess.close()
