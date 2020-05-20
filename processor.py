# _*_ coding: utf-8 _*_
# @Time : 2020/5/20 上午10:05 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : processor.py

import collections
import pickle
import re
import random

import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_window', 1, 'How many words to consider left and right.')
flags.DEFINE_integer('num_skips', 2, 'How many times to reuse an input to generate a label.')
flags.DEFINE_integer('n_words', 50000, 'max vocabulary size')


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
        f.close()
    return raw_word_list


def build_dict(file_in, file_out):

    op = open(file_out, 'wb')
    """Process raw inputs into a dataset."""
    words = read_data(file_in)
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(FLAGS.n_words - 1))

    word2id = {word: index for index, (word, _) in enumerate(count)}

    data = []
    unk_count = 0
    for word in words:
        index = word2id.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    id2word = dict(zip(word2id.values(), word2id.keys()))

    # 以pickle二进制文件存储
    pickle.dump(word2id, op)
    pickle.dump(id2word, op)
    pickle.dump(count, op)
    op.close()


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def convert_by_vocab(vocab, items, unk=0):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        # TODO: modify for oov, using [unk] replace, if you using english language do not change this
        # output.append(vocab.[item])
        # 100 对应 UNKNOW
        output.append(vocab.get(item, unk))
    return output


def dev(dict_file, file_out1, file_out2):
    dict_fp = open(dict_file, 'rb')
    word2id = pickle.load(dict_fp)
    id2word = pickle.load(dict_fp)

    op1 = open(file_out1, 'w', encoding='utf-8')
    op2 = open(file_out2, 'wb')

    valid_size = 64  # Random set of words to evaluate similarity on.
    valid_window = 5000  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_words = convert_by_vocab(id2word, valid_examples, unk='UNK')

    op1.write('\n'.join(valid_words))
    pickle.dump(valid_examples, op2)

    op1.close()
    op2.close()
    dict_fp.close()


def data_transformer(file_in, dict_file, file_out):
    fp = open(file_in, 'r', encoding='utf-8')
    dict_fp = open(dict_file, 'rb')

    writer = tf.python_io.TFRecordWriter(file_out)
    word2id = pickle.load(dict_fp)
    total_written = 0

    lines = fp.readlines()

    for i, line in enumerate(lines):
        if i%10000 == 0:
            print(i)
        line = re.sub('\n', '', line)
        words = line.split(' ')

        if len(words) > 1:
            word_index = 0
            while word_index < len(words):
                for j in range(1, FLAGS.skip_window+1):
                    if word_index-j >= 0:
                        total_written += 1
                        input_id = [word2id.get(words[word_index], 0)]
                        label = [word2id.get(words[word_index-j], 0)]
                        features = collections.OrderedDict()
                        features["input_id"] = create_int_feature(input_id)
                        features["label"] = create_int_feature(label)
                        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                        writer.write(tf_example.SerializeToString())

                    if word_index+j < len(words):
                        total_written += 1
                        input_id = [word2id.get(words[word_index], 0)]
                        label = [word2id.get(words[word_index+j], 0)]
                        features = collections.OrderedDict()
                        features["input_id"] = create_int_feature(input_id)
                        features["label"] = create_int_feature(label)
                        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                        writer.write(tf_example.SerializeToString())
                word_index += 1

    print("Wrote %d total instances", total_written)
    fp.close()
    dict_fp.close()
    writer.close()


if __name__ == '__main__':
    '''
    '''
    # file_in = './data/gzzf.txt'
    # file_out = './data/dict.pkl'
    # build_dict(file_in, file_out)

    # file_in = './data/test.txt'
    # dict_file = './data/dict.pkl'
    # file_out = './data/test.tf_record'
    # data_transformer(file_in, dict_file, file_out)


    dict_file = './data/dict.pkl'
    file_out1 = './data/dev.txt'
    file_out2 = './data/dev.pkl'
    dev(dict_file, file_out1, file_out2)