# _*_ coding: utf-8 _*_
# @Time : 2020/5/19 上午10:37 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : utils.py
import collections
import pickle
import random
import re


import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


def load_stopwords(stopwords_file):
    stop_words = set()
    fp = open(stopwords_file, 'r', encoding='utf-8')
    lines = fp.readlines()
    for line in lines:
        stop_words.add(line.strip())
    fp.close()
    return stop_words


def read_data(file_in):
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    # 读取文本，预处理，分词，得到词典
    raw_word_list = []
    with open(file_in, "r", encoding='UTF-8') as f:
        lines = f.readlines()
        if lines:
            for line in lines:
                line = re.sub('\n', '', line)
                words = line.split(' ')
                raw_word_list.extend(words)
    return raw_word_list


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = {word: index for index, (word, _) in enumerate(count)}
    data = []
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data_index = 0


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index - span) % len(data)
    return batch, labels


# 该函数用于统计 TFRecord 文件中的样本数量(总数)
def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return sample_nums


def reader_tfrecord(file_name, batch_size, capacity, min_after_dequeue):
    name_to_features = {
        "input_id": tf.FixedLenFeature([], tf.int64),
        "label": tf.FixedLenFeature([], tf.int64),
    }
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=name_to_features)
    input_id = tf.cast(features['input_id'], tf.int64)
    label = tf.cast(features['label'], tf.int64)

    input_ids_batch, labels_batch = tf.train.shuffle_batch([input_id, label], batch_size=batch_size
                                                           , capacity=capacity, min_after_dequeue=min_after_dequeue
                                                           , allow_smaller_final_batch=False)

    total_num = total_sample(file_name)
    data_batch = {}
    data_batch['input_ids'] = input_ids_batch
    data_batch['labels'] = labels_batch
    data_batch['total_num'] = total_num

    return data_batch


def load_dict(file_in):
    fp = open(file_in, 'rb')
    word2id = pickle.load(fp)
    id2word = pickle.load(fp)

    fp.close()
    return word2id, id2word


def load_valid_examples(file_in):
    fp = open(file_in, 'rb')
    valid_examples = pickle.load(fp)

    fp.close()
    return valid_examples


def plot_with_labels(low_dim_embs, labels, filename, fonts=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     fontproperties=fonts,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename, dpi=800)


def plot(final_embeddings, reverse_dictionary, filename):
    # 为了在图片上能显示出中文
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels, filename, fonts=None)
