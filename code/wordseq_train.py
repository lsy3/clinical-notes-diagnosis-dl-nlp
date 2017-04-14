import pandas as pd
import numpy as np
import cPickle
import argparse
import os, sys
import tensorflow as tf
from os.path import join
import wordseq_models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
from keras.utils import plot_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile',      dest='datafile', help='input pickle file', default='./data/DATA_WORDSEQV0_HADM_TOP10.p', type = str)
    parser.add_argument('--embmatrix', dest='embmatrix', help='embedding matrix', default='./data/EMBMATRIX_WORD2VEC_v2_100dim.p', type = str)
    parser.add_argument('--epoch',      dest='nb_epoch', help='number of epoch', default=50, type = int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=128, type = int)
    parser.add_argument('--model_name', dest='model_name', help='model loaded from *_model.py', default='conv1d_1', type=str)
    parser.add_argument('--pre_train_append', dest='pre_train_append', help='load weights_model_name<pre_train_append>', default='', type=str)
    parser.add_argument('--pre_train', dest = 'pre_train', help='continue train from pretrained para? True/False', default=False)
    parser.add_argument('--gpu', dest = 'gpu', help='set gpu no to be used (default: all)', default='',type=str)
    parser.add_argument('--plot_model', dest = 'plot_model', help='plot the said model', default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        print ('Use Default Settings ......')

    args = parser.parse_args()
    return args


def batch_generator(X, y, batch_size, shuffle, feature_size):
    number_of_batches = len(X)/batch_size
    counter = 0
    sample_index = np.arange(len(X))
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        y_batch = y[batch_index]
        X_batch = np.zeros((batch_size, feature_size))
        for i in range(batch_size):
            for j in X[i]:
                X_batch[i, j[0]] = j[1]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
    # return X_batch, y_batch


def sparse2dense(data, feature_size):
    dense_data = np.zeros((len(data), feature_size))
    for i in range(len(data)):
        for j in data[i]:
            dense_data[i, j[0]] = j[1]
    return dense_data


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

    f = open(args.embmatrix)
    embedding_matrix = cPickle.load(f)
    f.close()

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

    if args.plot_model:
        plot_model(model, args.plot_model, True, False)
        return

    if not os.path.isdir('./data/cache'):
        os.mkdir('./data/cache')
    weight_name = 'weights_' + model_name + args.pre_train_append + '.h5'
    weights_path = join('./data/cache', weight_name)
    if pre_train:
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


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train(args)
