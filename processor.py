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
flags.DEFINE_integer('skip_window', 3, 'How many words to consider left and right.')
flags.DEFINE_integer('num_skips', 3, 'How many times to reuse an input to generate a label.')
flags.DEFINE_integer('n_words', 100000, 'max vocabulary size')
flags.DEFINE_integer('t', 100000, 'subsampling threshold')


def read_data(file_in):
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    # 读取文本，预处理，分词，得到词典
    raw_word_list = []
    with open(file_in, "r", encoding='UTF-8') as f:
        count = 0
        while True:
            count += 1

            if count % 1000000 == 0:
                print(count)

            line = f.readline()
            if line:
                line = re.sub('\n', '', line)
                line = re.sub('\r', '', line)
                words = line.split(' ')
                raw_word_list.extend(words)
            else:
                break
        f.close()
    return raw_word_list


def build_bigdict(file_in, file_out):
    fp = open(file_in, "r", encoding='utf-8')
    op = open(file_out, 'wb')
    words_count = {
        'UNK': 10000000
    }
    count = 0
    while True:
        count += 1

        if count % 500000 == 0:
            print(count)

        line = fp.readline()
        if line:
            line = re.sub('\n', '', line)
            line = re.sub('\r', '', line)
            words = line.split(' ')
            if words:
                for word in words:
                    if word:
                        if words_count.__contains__(word):
                            words_count.update({word: words_count.get(word) + 1})
                        else:
                            words_count.update({word: 1})
        else:
            break

    words_count = sorted(words_count.items(), key=lambda x: x[1], reverse=True)
    words_count_len = len(words_count)
    if words_count_len > FLAGS.n_words:
        words_count = words_count[:FLAGS.n_words]

    word2id = {word: index for index, (word, _) in enumerate(words_count)}
    id2word = dict(zip(word2id.values(), word2id.keys()))
    words_count = dict(words_count)

    # 以pickle二进制文件存储
    pickle.dump(word2id, op)
    pickle.dump(id2word, op)
    pickle.dump(words_count, op)
    print(word2id)
    fp.close()
    op.close()


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
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_words = convert_by_vocab(id2word, valid_examples, unk='UNK')

    op1.write('\n'.join(valid_words))
    pickle.dump(valid_examples, op2)

    op1.close()
    op2.close()
    dict_fp.close()


def bigdata_transformer(file_in, dict_file, output_files, vocab_file=None):
    fp = open(file_in, 'r', encoding='utf-8')
    dict_fp = open(dict_file, 'rb')

    # writer = tf.python_io.TFRecordWriter(file_out)

    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    word2id = pickle.load(dict_fp)
    id2word = pickle.load(dict_fp)
    words_count = pickle.load(dict_fp)
    words_count.update({'UNK': 1e7})
    UNK_ID = word2id.get('UNK')

    if vocab_file:
        vocab_fp = open(vocab_file, 'w', encoding='utf-8')
        for key, value in word2id.items():
            vocab_fp.write(key)
            vocab_fp.write('\n')

    total_written = 0
    context_words = [j for j in range(1, FLAGS.skip_window + 1)]
    context_words.extend([-j for j in range(1, FLAGS.skip_window + 1)])
    count = 0
    while True:

        count += 1

        if count < 13000000:
            continue

        if count % 10000 == 0:
            print('读入文本行数%d' % count)

        line = fp.readline()
        if line:
            line = re.sub('\n', '', line)
            line = re.sub('\r', '', line)
            words = line.split(' ')

            if len(words) > 1:
                word_index = 0
                while word_index < len(words):
                    word = words[word_index]
                    if (word):
                        termfreq = words_count.get(word, 1e7)
                        random_p = random.random()
                        words_to_use = []

                        if termfreq > FLAGS.t:
                            discard_p = discard(termfreq)
                            if random_p > discard_p:
                                words_to_use = random.sample(context_words, FLAGS.num_skips)
                        else:
                            words_to_use = random.sample(context_words, FLAGS.num_skips)

                        for j in range(len(words_to_use)):

                            if (word_index + words_to_use[j] >= 0 and
                                    word_index + words_to_use[j] < len(words)):
                                total_written += 1
                                input_id = [word2id.get(word, UNK_ID)]
                                label = [word2id.get(words[word_index + words_to_use[j]], UNK_ID)]
                                # input_id = [words[word_index]]
                                # label = [words[word_index + words_to_use[j]]]
                                features = collections.OrderedDict()
                                features["input_id"] = create_int_feature(input_id)
                                features["label"] = create_int_feature(label)
                                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                                # writer.write(tf_example.SerializeToString())
                                writers[writer_index].write(tf_example.SerializeToString())
                                writer_index = (writer_index + 1) % len(writers)

                            if total_written % 1000000 == 0:
                                print('写入样例个数：', total_written)

                    word_index += 1
        else:
            break

    print("Wrote %d total instances" % total_written)
    fp.close()
    dict_fp.close()
    for writer in writers:
        writer.close()


def data_transformer(file_in, dict_file, file_out):
    fp = open(file_in, 'r', encoding='utf-8')
    dict_fp = open(dict_file, 'rb')

    writer = tf.python_io.TFRecordWriter(file_out)
    word2id = pickle.load(dict_fp)
    total_written = 0

    lines = fp.readlines()
    context_words = [j for j in range(1, FLAGS.skip_window + 1)]
    context_words.extend([-j for j in range(1, FLAGS.skip_window + 1)])

    for i, line in enumerate(lines):
        if i % 10000 == 0:
            print(i)
        line = re.sub('\n', '', line)
        words = line.split(' ')

        if len(words) > 1:
            word_index = 0
            while word_index < len(words):

                words_to_use = random.sample(context_words, FLAGS.num_skips)
                for j in range(len(words_to_use)):

                    if (word_index + words_to_use[j] >= 0 and
                            word_index + words_to_use[j] < len(words)):
                        total_written += 1
                        input_id = [word2id.get(words[word_index], 0)]
                        label = [word2id.get(words[word_index + words_to_use[j]], 0)]
                        features = collections.OrderedDict()
                        features["input_id"] = create_int_feature(input_id)
                        features["label"] = create_int_feature(label)
                        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                        writer.write(tf_example.SerializeToString())
                        if total_written % 10000 == 0:
                            print('写入样例个数：', total_written)

                word_index += 1

    print("Wrote %d total instances" % total_written)
    fp.close()
    dict_fp.close()
    writer.close()


def discard(termfreq):
    p = FLAGS.t / termfreq
    p = p ** 0.5
    p = 1 - p
    return p


if __name__ == '__main__':
    '''
    '''
    # file_in = './data/v0.0.1/gzzf_remove_digit2.txt'
    # file_out = './data/v0.0.1/dict.pkl'
    # build_bigdict(file_in, file_out)

    # file_in = './data/v0.0.1/gzzf_remove_digit2.txt'
    # dict_file = './data/v0.0.1/dict.pkl'
    # # file_out = './data/v0.0.1/data1.tf_record'
    # output_files = ['./data/v0.0.1/data2.tf_record',
    #                 ]
    # vocab_file = './data/v0.0.1/vocab.txt'
    # bigdata_transformer(file_in, dict_file, output_files, vocab_file)

    # dict_file = './data/v0.0.1/dict.pkl'
    # file_out1 = './data/v0.0.1/dev.txt'
    # file_out2 = './data/v0.0.1/dev.pkl'
    # dev(dict_file, file_out1, file_out2)

    print(FLAGS.t)
