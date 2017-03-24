# Simple Word2Vec
# Input: read training and testing data from './data' file,
# Output: save the embedding matrix to './data' folder with h5 format

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

import collections
import math
import os
import random
import h5py
import os
import time
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import matplotlib.pyplot as plt


def build_dataset(words, vocabulary_size):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary


def generate_batch(data, data_index, batch_size, num_skips, skip_window):
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels, data_index


def embedding_training(index_list, data_index, reverse_dictionary):
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    vocabulary_size = len(index_list) + 1
    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64  # Number of negative examples to sample.

    graph = tf.Graph()

    # construct tensorflow graph
    with graph.as_default():

        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()

    num_steps = 2000

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels, data_index = generate_batch(
                index_list, data_index, batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(1, valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    print(reverse_dictionary[valid_examples[i]])
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()
    return final_embeddings


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


def save_to_h5(file_path, embedding, dictionary):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'embedding_matrix_' + timestr + '.h5'
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    full_name = os.path.join(file_path, file_name)

    with h5py.File(full_name, 'w') as hf:
        hf.create_dataset('embedding_matrix', data=embedding)
        hf.create_dataset('dictionary_keys', data=dictionary.keys())
        hf.create_dataset('dictionary_values', data = dictionary.values())
    print('data saved!')



if __name__ == "__main__":
    data_df = pd.read_csv('./data/data.tsv', sep='\t', header=0)
    data_values = data_df['Phrase'].values
    print('data size: ' + str(data_df.shape[0]))

    # top n codes
    n = 10
    # random feeding some value to the label
    label = np.random.rand(data_values.shape[0], n)
    label[label < 0.5] = 0
    label[label >= 0.5] = 1

    from sklearn.model_selection import train_test_split
    train_data, test_data, train_label, test_label = train_test_split(data_values, label, test_size = 0.20, random_state=42)


    from keras.preprocessing.text import Tokenizer
    toke = Tokenizer()
    toke.fit_on_texts(train_data)
    dictionary = toke.word_index
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    index_list = dictionary.values()

    train_sequence = toke.texts_to_sequences(train_data)
    test_sequence = toke.texts_to_sequences(test_data)

    import cPickle
    f = open('./data/preprocessing_data.p', 'wb')
    for obj in [reverse_dictionary, train_sequence, test_sequence,
                train_label, test_label]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    # Function to generate a training batch for the skip-gram model.
    data_index = 0
    batch, labels, data_index = generate_batch(index_list, data_index, batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
      print(batch[i], reverse_dictionary[batch[i]],
            '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    # Train word2vec and get the embedding matrix
    vocabulary_size = 5000
    embedding_matrix = embedding_training(index_list, data_index, reverse_dictionary)

    file_path = './data'
    # timestr = time.strftime("%H%M%S")
    file_name = 'embedding_matrix.p'
    # file_name = 'embedding_matrix-' + timestr + '.p'
    full_name = os.path.join(file_path, file_name)
    cPickle.dump(embedding_matrix, open(full_name, 'wb'))
    print('data saved!')

    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(embedding_matrix[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(1, plot_only)]
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print("Please install sklearn and matplotlib to visualize embeddings")

