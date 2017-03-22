from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import h5py


def restore_h5(file_name):
    hf = h5py.File(file_name, 'r')
    embedding_matrix = np.copy(hf['embedding_matrix'])
    dictionary = np.copy(hf['dictionary'])
    hf.close()
    return embedding_matrix, dictionary



if __name__ == "__main__":
    # Phase 1: read data to df
    train_df = pd.read_csv('./data/train.tsv', sep='\t', header=0)
    test_df = pd.read_csv('./data/test.tsv', sep='\t', header=0)

    train_x = train_df['Phrase'].values
    test_x = test_df['Phrase'].values
    print('train size: ' + str(train_x.shape[0]) + ' test size: ' + str(test_x.shape[0]))

    train_y = train_df['Sentiment'].values
    print('number of labels: ' + str(len(np.unique(train_y))))

    # TODO: Should read sequence from our saved file, the keras Tokenizer here is just a placeholder.
    # The ids in sequence should match the ids in dictionary
    toke = Tokenizer()
    toke.fit_on_texts(train_x)
    sequences = toke.texts_to_sequences(train_x)


    from keras.utils import np_utils

    MAX_SEQUENCE_LENGTH = 20
    EMBEDDING_DIM = 100

    X_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # -- TODO -- multi label y_train
    y_train = np_utils.to_categorical(train_df["Sentiment"].values)

    embedding_matrix, dictionary = restore_h5('./data/embedding_matrix_20170322-213558.h5')


    #LSTM with embedding trainable
    from keras.models import Model
    from keras.layers import Input
    from keras.optimizers import Adam
    from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization
    from keras.layers import LSTM

    opt = Adam(0.002)
    inp = Input(shape=X_train.shape[1:])
    x = Embedding(len(dictionary) + 1,
                  EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_SEQUENCE_LENGTH,
                  trainable=True,
                  input_shape=X_train.shape[1:])(inp)
    x = LSTM(256, return_sequences=False, dropout_W = 0.3, dropout_U = 0.3)(x)
    x = BatchNormalization()(x)
    pred = Dense(5,activation='softmax')(x)

    model = Model(inp,pred)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    model.summary()



