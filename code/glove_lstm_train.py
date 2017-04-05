from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import cPickle
import os
import argparse
import sys
from os.path import join
import dl_models
from keras.callbacks import ModelCheckpoint, EarlyStopping

MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',      dest='nb_epoch', help='number of epoch', default=50, type = int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=128, type=int)
    parser.add_argument('--model_name', dest='model_name', help='model loaded from dl_model.py', default='lstm_model_1', type=str)
    parser.add_argument('--pre_train', dest = 'pre_train', help='continue train from pretrained para? True/False', default=False)
    if len(sys.argv) == 1:
        parser.print_help()
        print ('Use Default Settings ......')

    args = parser.parse_args()
    return args


def get_embedding_matrix(word_index):
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

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    found = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            found += 1
            embedding_matrix[i] = embedding_vector

    print('Using %i word vectors of total vocabulary size: %i' % (found, len(word_index)))
    return embedding_matrix


def train(data_file):
    args = parse_args()
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    model_name = args.model_name
    pre_train = args.pre_train

    f = open(data_file, 'rb')
    loaded_data = []
    for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        loaded_data.append(cPickle.load(f))
    f.close()

    dictionary = loaded_data[0]
    train_sequence = loaded_data[1]
    test_sequence = loaded_data[2]
    train_label = loaded_data[3]
    test_label = loaded_data[4]

    train_sequence = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    test_sequence = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)

    word_index = dictionary.values()
    embedding_matrix = get_embedding_matrix(word_index)


    function_list = dl_models.get_function_dict()
    model_func = function_list[model_name]

    vocabulary_size = len(dictionary) + 1
    category_number = train_label.shape[1]
    input_shape = train_sequence.shape[1:]

    model = model_func(vocabulary_size, embedding_matrix, MAX_SEQUENCE_LENGTH, category_number, input_shape)

    if not os.path.isdir('./data/cache'):
        os.mkdir('./data/cache')
    weight_name = 'weights_' + model_name + '.h5'
    weights_path = join('./data/cache', weight_name)
    if pre_train == True:
        model.load_weights(weights_path)
    print ('checkpoint')
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    model.summary()
    model.fit(train_sequence, train_label, validation_split= 0.2, epochs=nb_epoch, batch_size=batch_size,
              callbacks=[checkpointer, earlystopping])


if __name__ == "__main__":
    data_file = './data/preprocessing_data.p'
    train(data_file)