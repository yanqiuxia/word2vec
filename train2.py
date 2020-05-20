# _*_ coding: utf-8 _*_
# @Time : 2020/5/19 上午10:19 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : train.py
import os
import tensorflow as tf
import numpy as np

from model import Model
from utils import load_dict, load_valid_examples, plot, reader_tfrecord

input_file = './data/data.tf_record'
dev_file = './data/dev.pkl'
dict_file = './data/dict.pkl'

version = 'v0.0.1'
save_dir = os.path.join('./checkpoint', version)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'model')

log_dir = './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

image_file = './images/tsne3.png'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('vocabulary_size', 50000, 'vocabulary size')
flags.DEFINE_integer('embedding_size', 128, 'Dimension of the embedding vector. ')
flags.DEFINE_integer('num_sampled', 64, 'negative sample num')
flags.DEFINE_float('lr', 1.0, ' init learning rate')
flags.DEFINE_integer('skip_window', 2, 'How many words to consider left and right.')
flags.DEFINE_integer('num_skips', 4, 'How many times to reuse an input to generate a label.')
flags.DEFINE_integer('num_true', 1, 'Actual number of positive samples')

flags.DEFINE_integer('batch_size', 128, 'train bacth size')
flags.DEFINE_integer('valid_size', 16, 'Random set of words to evaluate similarity on.')
flags.DEFINE_integer('valid_window', 100, 'Only pick dev samples in the head of the distribution.')

flags.DEFINE_string('input_file', input_file, 'the input file path')
flags.DEFINE_string('dev_file', dev_file, 'the input file path')
flags.DEFINE_string('dict_file', dict_file, 'the dict file path')
flags.DEFINE_string('save_path', save_path, 'new model save path')
flags.DEFINE_string('log_dir', log_dir, 'lor dir path')
flags.DEFINE_string('image_file', image_file, 'result save file')

flags.DEFINE_boolean('is_train', True, 'whether to Training model or not')
flags.DEFINE_integer('num_steps', 10000, ' train num steps')
flags.DEFINE_integer('epoch', 50, 'training epoch')


class SGNS(object):
    def __init__(self, train_data, word2id, id2word, valid_examples):

        self.data = train_data
        self.dictionary = word2id
        self.reverse_dictionary = id2word
        self.valid_examples = valid_examples
        # Step 4: Build and train a skip-gram model.

        self.model = Model(vocabulary_size=len(self.dictionary),
                           embedding_size=FLAGS.embedding_size,
                           num_sampled=FLAGS.num_sampled,
                           lr=FLAGS.lr,
                           valid_examples=self.valid_examples,
                           num_true=FLAGS.num_true)

    def train(self):
        '''

        :return:
        '''

        train_batch_num = int(self.data['total_num'] / FLAGS.batch_size)


        self.sess.run(tf.global_variables_initializer())
        # Create a saver.
        saver = tf.train.Saver()
        # Merge all summaries.
        merged = tf.summary.merge_all()
        print('init ')
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(FLAGS.log_dir, self.sess.graph)
        average_loss = 0
        for i in range(FLAGS.epoch):
            for j in range(train_batch_num):
                step = i * train_batch_num + j
                # batch_inputs, batch_labels = generate_batch(self.data, FLAGS.batch_size, FLAGS.num_skips,
                #                                             FLAGS.skip_window)
                input_ids = self.data['input_ids']
                labels = self.data['labels']
                batch_inputs, batch_labels = self.sess.run([input_ids, labels])
                batch_labels = batch_labels.reshape([FLAGS.batch_size, 1])

                feed_dict = {
                        self.model.train_inputs: batch_inputs,
                        self.model.train_labels: batch_labels
                    }

                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned
                # "summary" variable. Feed metadata variable to session for visualizing
                # the graph in TensorBoard.
                _, summary, loss_val = self.sess.run([self.model.train_op, merged, self.model.loss],
                                                feed_dict=feed_dict,
                                                run_metadata=run_metadata)
                average_loss += loss_val

                # Add returned summaries to writer in each step.
                writer.add_summary(summary, step)
                # Add metadata to visualize the graph for the last run.
                if step == (FLAGS.num_steps - 1):
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000
                    # batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if (step + 1) % 10000 == 0:
                    sim = self.model.similarity.eval()
                    for k in range(FLAGS.valid_size):
                        valid_word = self.reverse_dictionary[self.valid_examples[k]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[k, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word

                        print(log_str, ', '.join([self.reverse_dictionary[nearest[k]] for k in range(top_k)]))
        final_embeddings = self.model.normalized_embeddings.eval()
        # Save the model for checkpoints.
        saver.save(self.sess, FLAGS.save_path)
        plot(final_embeddings, self.reverse_dictionary, FLAGS.image_file)

        writer.close()
        self.sess.close()

    def creat_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)

    def close_session(self):
        # 请求线程结束
        self.coord.request_stop()
        # 等待线程终止
        self.coord.join(self.threads)
        self.sess.close()

    def evaluate(self):
        '''

        :return:
        '''


def main(_):
    # step 1:读取文件中的内容组成一个列表
    train_data = reader_tfrecord(FLAGS.input_file, FLAGS.batch_size
                                 , capacity=64, min_after_dequeue=10)

    # Step 2: Build the dictionary and replace rare words with UNK token.
    word2id, id2word = load_dict(FLAGS.dict_file)

    # step 3： valid_example
    valid_examples = load_valid_examples(FLAGS.dev_file)

    sgns = SGNS(train_data=train_data,
                word2id=word2id,
                id2word=id2word,
                valid_examples=valid_examples)
    sgns.creat_session()

    if FLAGS.is_train:
        print('begin training!')
        sgns.train()
        print('Model training completed!')
    else:
        sgns.evaluate()

    sgns.close_session()

if __name__ == '__main__':
    tf.app.run()