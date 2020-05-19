# _*_ coding: utf-8 _*_
# @Time : 2020/5/19 上午10:19 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : train.py
import os
import tensorflow as tf
import numpy as np

from model import Model
from utils import read_data, build_dataset, generate_batch, plot

input_file = './data/gzzf.txt'
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
flags.DEFINE_string('save_path', save_path, 'new model save path')
flags.DEFINE_string('log_dir', log_dir, 'lor dir path')
flags.DEFINE_string('image_file', image_file, 'result save file')

flags.DEFINE_boolean('is_train', True, 'whether to Training model or not')
flags.DEFINE_integer('num_steps', 10000, ' train num steps')
flags.DEFINE_integer('epoch', 50, 'training epoch')


class SG(object):
    def __init__(self, data, dictionary, reverse_dictionary, valid_examples):

        self.data = data
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.valid_examples = valid_examples
        # Step 4: Build and train a skip-gram model.
        self.graph = tf.Graph()
        with self.graph.as_default():
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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        train_batch_num = int(len(self.data) / FLAGS.batch_size)

        # Step 5: Begin training.
        with tf.Session(config=config, graph=self.graph) as sess:

            sess.run(tf.global_variables_initializer())
            # Create a saver.
            saver = tf.train.Saver()
            # Merge all summaries.
            merged = tf.summary.merge_all()
            print('init ')
            # Open a writer to write summaries.
            writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            average_loss = 0
            for i in range(FLAGS.epoch):
                for j in range(train_batch_num):
                    step = i*train_batch_num+j
                    batch_inputs, batch_labels = generate_batch(self.data, FLAGS.batch_size, FLAGS.num_skips,
                                                                FLAGS.skip_window)
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
                    _, summary, loss_val = sess.run([self.model.train_op, merged, self.model.loss],
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
                    if (step+1) % 10000 == 0:
                        sim = self.model.similarity.eval()
                        for k in range(FLAGS.valid_size):
                            valid_word = self.reverse_dictionary[self.valid_examples[k]]
                            top_k = 8  # number of nearest neighbors
                            nearest = (-sim[k, :]).argsort()[1:top_k + 1]
                            log_str = 'Nearest to %s:' % valid_word

                            print(log_str, ', '.join([self.reverse_dictionary[nearest[k]] for k in range(top_k)]))
            final_embeddings = self.model.normalized_embeddings.eval()
            # Save the model for checkpoints.
            saver.save(sess, FLAGS.save_path)
            plot(final_embeddings, self.reverse_dictionary, FLAGS.image_file)

        writer.close()
        sess.close()

    def evaluate(self):
        '''

        :return:
        '''


def main(_):
    # step 1:读取文件中的内容组成一个列表
    words = read_data(FLAGS.input_file)
    print('Data size', len(words))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    data, count, dictionary, reverse_dictionary = build_dataset(words, FLAGS.vocabulary_size)
    del words  # 删除words节省内存
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    # We pick a random validation set to sample nearest neighbors. Here we limit
    # the validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    sg = SG(data=data,
            dictionary=dictionary,
            reverse_dictionary=reverse_dictionary,
            valid_examples=valid_examples)
    if FLAGS.is_train:
        print('begin training!')
        sg.train()
        print('Model training completed!')
    else:
        sg.evaluate()


if __name__ == '__main__':
    tf.app.run()
