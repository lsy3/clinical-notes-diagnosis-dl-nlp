import pandas as pd
import numpy as np
import cPickle
import argparse
import os, sys
from os.path import join
import wordseq_models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding


MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 300


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile',      dest='datafile', help='input pickle file', default='./data/DATA_WORDSEQV0_HADM_TOP10.p', type = str)
    parser.add_argument('--embmatrix', dest='embmatrix', help='embedding matrix', default='./data/EMBMATRIX_WORD2VEC_v2_100dim.p', type = str)
    parser.add_argument('--epoch',      dest='nb_epoch', help='number of epoch', default=50, type = int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=128, type = int)
    parser.add_argument('--model_name', dest='model_name', help='model loaded from *_model.py', default='conv2d_1', type=str)
    parser.add_argument('--pre_train', dest = 'pre_train', help='continue train from pretrained para? True/False', default=False)
    if len(sys.argv) == 1:
        parser.print_help()
        print ('Use Default Settings ......')

    args = parser.parse_args()
    return args


def get_embedding_matrix(dictionary):
    # load glove vectors
    # download them from http://nlp.stanford.edu/data/glove.6B.zip
    embeddings_index = {}
    GLOVE_DIR = './data'
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.%id.txt' % EMBEDDING_DIM))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    # take only word vecs that are in training dictionary

    embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
    found = 0
    for word, i in dictionary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            found += 1
            embedding_matrix[i] = embedding_vector

    print('Using %i word vectors of total vocabulary size: %i' % (found, len(dictionary)))
    return embedding_matrix


def train(args):
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    model_name = args.model_name
    pre_train = args.pre_train

    f = open(args.datafile, 'rb')
    loaded_data = []
    for i in range(7): # [reverse_dictionary, train_sequence, test_sequence, train_label, test_label]:
        loaded_data.append(cPickle.load(f))
    f.close()

    dictionary = loaded_data[0]
    train_sequence = loaded_data[1]
    val_sequence = loaded_data[2]
    test_sequence = loaded_data[3]
    train_label = loaded_data[4]
    val_label = loaded_data[5]
    test_label = loaded_data[6]

    embedding_matrix = get_embedding_matrix(dictionary)

    max_sequence_length = train_sequence.shape[1]
    vocabulary_size = len(dictionary) + 1
    embedding_dim = embedding_matrix.shape[1]
    category_number = train_label.shape[1]
    input_shape = train_sequence.shape[1:]

    embedding_layer = Embedding(vocabulary_size,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_sequence_length,
                        trainable=False,
                        input_shape=input_shape)

    model_func = getattr(wordseq_models, model_name)
    model = model_func(input_shape, category_number, embedding_layer)

    if not os.path.isdir('./data/cache'):
        os.mkdir('./data/cache')
    weight_name = 'weights_' + model_name + '.h5'
    weights_path = join('./data/cache', weight_name)
    if pre_train == True:
        model.load_weights(weights_path)
    print ('checkpoint')
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    #train_sequence = np.concatenate((train_sequence, val_sequence), axis=0)
    #train_label = np.concatenate((train_label, val_label), axis=0)

    model.fit(train_sequence, train_label,
              batch_size = batch_size,
              epochs = nb_epoch,
              validation_data = [val_sequence, val_label],
              callbacks=[checkpointer, earlystopping])


if __name__ == "__main__":
    args = parse_args()
    train(args)