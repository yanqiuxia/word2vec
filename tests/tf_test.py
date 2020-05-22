# _*_ coding: utf-8 _*_
# @Time : 2020/5/19 上午10:24 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : tf_test.py
import tensorflow as tf

from utils import reader_tfrecord

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, FontManager
import matplotlib.pyplot as plt
import subprocess


input_file = '../data/data.tf_record'

# !/usr/bin/python
# coding:utf-8
import tensorflow as tf
import numpy as np

# images = np.random.random([5, 2])
# label = np.asarray(range(0, 5))
# images = tf.cast(images, tf.float32)
# label = tf.cast(label, tf.int32)
# input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
# # 将队列中数据打乱后再读取出来
# image_batch, label_batch = tf.train.shuffle_batch(input_queue, batch_size=10, num_threads=1, capacity=64,
#                                                   min_after_dequeue=1)

# with tf.Session() as sess:
#     # 线程的协调器
#     coord = tf.train.Coordinator()
#     # 开始在图表中收集队列运行器
#     threads = tf.train.start_queue_runners(sess, coord)
#     image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
#     for j in range(5):
#         # print(image_batch_v.shape, label_batch_v[j])
#         print(image_batch_v[j]),
#         print(label_batch_v[j])
#
#     data = reader_tfrecord(input_file, 6
#                            , capacity=64, min_after_dequeue=10)
#
#     input_ids = data['input_ids']
#     labels = data['labels']
#     batch_inputs, batch_labels = sess.run([input_ids, labels])
#     # 请求线程结束
#     coord.request_stop()
#     # 等待线程终止
#     coord.join(threads)

# x = np.random.random([5, 2])
# x = tf.cast(x, tf.float32)
#
# norm = tf.norm(x, axis=1, keep_dims=True)
# b = x/norm
# with tf.Session() as sess:
#     print(x.eval())
#     print(sess.run(norm))
#     print(sess.run(b))



def dispFonts():
    #显示可用的中文字体，同时支持英文的
    from matplotlib.font_manager import FontManager
    import subprocess

    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)

    output = subprocess.check_output(
        'fc-list :lang=zh -f "%{family}\n"', shell=True)
    output = output.decode('utf-8')
    # print '*' * 10, '系统可用的中文字体', '*' * 10
    # print output
    zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
    available = mat_fonts & zh_fonts

    print ('*' * 10 +  u'可用的中文字体'+'*' * 10)
    for f in available:
        print(f)
        #/ usr / share / fonts / truetype / droid / DroidSansFallbackFull.ttf

if __name__ == '__main__':

    dispFonts()
    myfont = FontProperties(fname='/usr/share/fonts/MyFonts/YaHei.Consolas.1.11b.ttf', size=20)
    # rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体，但在下运行的时候报了warning并没正常显示中文
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot([1, 2, 3], [1, 2, 3], 'bv-')
    ax1.set_xlabel(u'x轴标签', fontproperties=myfont)
    ax1.set_ylabel(u'y轴标签', fontproperties=myfont)
    plt.show()
    ax1.legend([u'图例标签'], loc='best', prop=myfont)