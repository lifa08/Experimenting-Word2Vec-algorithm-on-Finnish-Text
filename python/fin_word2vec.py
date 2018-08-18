import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

import fin_w2v_utils

# model hyperparameters
EMBED_SIZE = 128 # dimension of the word embedding vectors
NUM_SAMPLED = 64 # number of negative examples to sample
LEARNING_RATE = 1.0
SKIP_STEP = 1000
NUM_TRAIN_STEPS = 50000

VOCAB_SIZE = 5000
BATCH_SIZE = 128
NUM_VISUALIZE = 3000 # number of tokens to visualize
SKIP_WINDOW = 1 # the context window

SCRIPT_DIR = os.path.dirname(os.path.abspath('__file__'))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DATA_FILE_NAME = 'europarl-v8.fi.tgz'
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

VOCAB_DIR = os.path.join(PROJECT_DIR, 'vocab')

# Parameters for downloading data
DOWNLOAD_URL = 'http://statmt.org/wmt15/europarl-v8.fi.tgz'
EXPECTED_BYTES = 99540237

class SkipGramModel:
    ''' Build the graph for word2vec model '''
    def __init__(self, dataset, vocab_size, embed_size, batch_size,
                 num_sampled, learning_rate, skip_step, chkpt_dir, graph_dir,
                 graph):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self_batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.skip_step = skip_step
        self.chkpt_dir = chkpt_dir
        self.graph_dir = graph_dir
        self.graph = graph

        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step',
                                               initializer=tf.constant(0),
                                               trainable=False)
            self.dataset = dataset

    def _import_data(self):
        ''' Step 1: import data '''
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words=self.iterator.get_next()

    def _create_embedding(self):
        ''' Step 2 + 3: define weights and embedding lookup.
        In word2vec, it is actually the weights that we care about
        '''
        with tf.name_scope('embed'):
            self.embed_matrix = tf.get_variable('embed_matrix',
                                               shape=[self.vocab_size, self.embed_size],
                                               initializer=tf.random_uniform_initializer())
            # default range [0, 1] for float datatype fro tf.random_uniform_initializer
            self.embed = tf.nn.embedding_lookup(self.embed_matrix,
                                           self.center_words, name='embedding')

    def _create_loss(self):
        ''' Step 4: define the loss function '''
        with tf.name_scope('loss'):
            # construct variables for NCE loss
            # nce_weight and bias are the weights and biases for the last softmax layer
            nce_weight = tf.get_variable('nce_weight', 
                                        shape=[self.vocab_size, self.embed_size],
                                        initializer=tf.truncated_normal_initializer(
                                            stddev=1.0/(self.embed_size ** 0.5)))

            nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([self.vocab_size]))

            # define loss function to be NCE loss function
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                     biases=nce_bias,
                                                     labels=self.target_words,
                                                     inputs=self.embed,
                                                     num_sampled=self.num_sampled,
                                                     num_classes=self.vocab_size),
                                        name='loss')
    def _create_optimizer(self):
        ''' Step 5: define optimizer'''
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                             global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        ''' Build the graph for our model '''
        with self.graph.as_default():
            self._import_data()
            self._create_embedding()
            self._create_loss()
            self._create_optimizer()
            self._create_summaries()

    def train(self, num_train_steps):
        # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias
        with self.graph.as_default():
            saver = tf.train.Saver()

        initial_step = 0
        fin_w2v_utils.safe_mkdir(self.chkpt_dir)
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.chkpt_dir)

            # if that checkpoint exists, restore from checkporint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('restoing from checkpoint')

            # we use this to calculate late average loss in the last SKIP_STEP steps
            total_loss = 0.0
            writer = tf.summary.FileWriter(self.graph_dir + str(self.lr), sess.graph)
            initial_step = self.global_step.eval()

            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    loss_batch, _, summary = sess.run([self.loss,
                                                       self.optimizer,
                                                       self.summary_op])

                    # Need global step here so the model knows
                    # what summary corresponds to what step
                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch

                    if (index + 1) % self.skip_step == 0:
                        print('Average loss at step {}:{:5.2f}'
                              .format(index, total_loss/self.skip_step))

                        total_loss = 0.0
                        saver.save(sess, self.chkpt_dir + '/skip-gram', index)

                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)

            writer.close()

    def visualize(self, vocab_file_path, visual_dir, num_visualize):
        ''' run tensorboad --logdir=visualization to see the embeddings '''

        # write the num_visualize most common words into file '/visual_dir/vocab_str(num_visualize)+.tsv'
        fin_w2v_utils.most_common_words(vocab_file_path, visual_dir, num_visualize)

        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.chkpt_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            final_embed_matrix = sess.run(self.embed_matrix)

            # you have to store embddings in a new variable
            embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter(visual_dir)

            # add embedding to the config file
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # link this tensor to its metadata file (labels), in this 
            # case the first NUM_VISUALIZE words of vocab
            embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

            # saves a configuation file that TensorBoard with read during startup.
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, os.path.join(visual_dir, 'model.ckpt'), 1)

def train_without_lemmatization():
    VISUAL_DIR = os.path.join(PROJECT_DIR, 'visualization/norm')
    VOCAB_FILE_NAME = 'fin_vocab_' + str(VOCAB_SIZE) + '.tsv'
    VOCAB_FILE_PATH = os.path.join(VOCAB_DIR, VOCAB_FILE_NAME)

    INDEX_FILE = os.path.join(DATA_DIR, 'sentence_idx_imdb_' + str(VOCAB_SIZE) + '.pkl')
    CHKPT_DIR = os.path.join(PROJECT_DIR, 'checkpoints/norm')
    GRAPH_DIR = os.path.join(PROJECT_DIR, 'graphs/norm/lr_')

    # Lemmatize or not
    LEMMATIZE = False

    GRAPH = tf.Graph()

    def gen():
        yield from fin_w2v_utils.batch_gen(DOWNLOAD_URL, DATA_DIR, DATA_FILE_PATH, EXPECTED_BYTES,
                                           VOCAB_DIR, VOCAB_FILE_PATH, VOCAB_SIZE, INDEX_FILE,
                                           BATCH_SIZE, SKIP_WINDOW, LEMMATIZE)

    with GRAPH.as_default():
        dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), 
                                                 (tf.TensorShape([BATCH_SIZE]),
                                                  tf.TensorShape([BATCH_SIZE, 1])))

    model = SkipGramModel(dataset, VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE,
                          NUM_SAMPLED, LEARNING_RATE, SKIP_STEP, CHKPT_DIR,
                          GRAPH_DIR, GRAPH)

    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
    model.visualize(VOCAB_FILE_PATH, VISUAL_DIR, NUM_VISUALIZE)

def train_with_lemmatization():
    # model hyperparameters
    VISUAL_DIR = os.path.join(PROJECT_DIR, 'visualization/lemmatized')
    VOCAB_FILE_NAME = 'fin_vocab_' + str(VOCAB_SIZE) + '_lemmatized.tsv'
    VOCAB_FILE_PATH = os.path.join(VOCAB_DIR, VOCAB_FILE_NAME)
    INDEX_FILE = os.path.join(DATA_DIR, 'sentence_idx_imdb_' + str(VOCAB_SIZE) + '_lemmatized.pkl')
    CHKPT_DIR = os.path.join(PROJECT_DIR, 'checkpoints/lemmatized')
    GRAPH_DIR = os.path.join(PROJECT_DIR, 'graphs/lemmatized/lr_')

    # Lemmatize or not
    LEMMATIZE = True

    GRAPH_LEMMATIZE = tf.Graph()

    def gen():
        yield from fin_w2v_utils.batch_gen(DOWNLOAD_URL, DATA_DIR, DATA_FILE_PATH, EXPECTED_BYTES,
                                           VOCAB_DIR, VOCAB_FILE_PATH, VOCAB_SIZE, INDEX_FILE,
                                           BATCH_SIZE, SKIP_WINDOW, LEMMATIZE)

    with GRAPH_LEMMATIZE.as_default():
        dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), 
                                                 (tf.TensorShape([BATCH_SIZE]),
                                                  tf.TensorShape([BATCH_SIZE, 1])))

    model_lemmatized = SkipGramModel(dataset, VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE,
                                     NUM_SAMPLED, LEARNING_RATE, SKIP_STEP, CHKPT_DIR,
                                     GRAPH_DIR, GRAPH_LEMMATIZE)
    
    model_lemmatized.build_graph()
    model_lemmatized.train(NUM_TRAIN_STEPS)
    model_lemmatized.visualize(VOCAB_FILE_PATH, VISUAL_DIR, NUM_VISUALIZE)

def main(lemmatize):
    if lemmatize:
        train_with_lemmatization()
    else:
        train_without_lemmatization()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a word2vec model using Finnish text')
    parser.add_argument('-l', '--lemmatize', type=bool, default=False,
                        help='lemmatize the text or not (default: False, not lemmatizing)')

    args = parser.parse_args()
    main(lemmatize=args.lemmatize)
