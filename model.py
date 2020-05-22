# _*_ coding: utf-8 _*_
# @Time : 2020/5/19 上午10:01 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : model.py
import tensorflow as tf
import math


class Model(object):

    def __init__(self, vocabulary_size, embedding_size, num_sampled, lr, valid_examples, num_true):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.lr = lr
        self.valid_examples = valid_examples
        self.num_true = num_true

        self.train()

    def nce_loss(self, embed, y_labels):
        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=y_labels,
                inputs=embed,
                num_true=self.num_true,
                num_sampled=self.num_sampled,
                num_classes=self.vocabulary_size))
        return loss

    def train(self):
        with tf.name_scope('inputs'):
            self.train_inputs = tf.placeholder(tf.int32, shape=[None]) #[batch_size]
            self.train_labels = tf.placeholder(tf.int32, shape=[None, self.num_true]) #[batch_size]
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name='word_embedding')
            embed_input = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

            dir_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name='dir_embedding')
            dir_input = tf.nn.embedding_lookup(dir_embeddings, self.train_inputs)

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss and why choosing NCE over tf.nn.sampled_softmax_loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            #   http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf
        with tf.name_scope('loss'):
            with tf.name_scope('vec'):
                vec_loss = self.nce_loss(embed_input, self.train_labels)
            with tf.name_scope('dir'):
                dir_loss = self.nce_loss(dir_input, self.train_labels)

            self.loss = vec_loss + dir_loss

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', self.loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(self.lr)

        self.train_op = optimizer.minimize(self.loss)

        var_list = tf.trainable_variables()
        for var in var_list:
            print("Variable: ", var)

        # Compute the cosine similarity between minibatch examples and all
        # embeddings.
        # norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        norm = tf.norm(self.embeddings, axis=1, keep_dims=True)
        # tf.Print(embeddings, [norm, norm.shape], summarize=10, message='norm:')
        self.normalized_embeddings = self.embeddings / norm
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True)
