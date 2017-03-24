from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
import cPickle


if __name__ == "__main__":
    f = open('./data/preprocessing_data.p', 'rb')
    loaded_data = []
    for i in range(5): # [reverse_dictionary, train_sequence, test_sequence, train_label, test_label]:
        loaded_data.append(cPickle.load(f))
    f.close()

    dictionary = loaded_data[0]
    train_sequence = loaded_data[1]
    test_sequence = loaded_data[2]
    train_label = loaded_data[3]
    test_label = loaded_data[4]

    max_sequence_length = 20
    train_sequence = pad_sequences(train_sequence, maxlen=max_sequence_length)
    test_sequence = pad_sequences(test_sequence, maxlen=max_sequence_length)
    f = open('./data/embedding_matrix.p')
    embedding_matrix = cPickle.load(f)
    f.close

    # Sequence classification with Convolution
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Embedding, BatchNormalization
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

    vocabulary_size = len(dictionary) + 1
    embedding_dim = embedding_matrix.shape[1]
    category_number = train_label.shape[1]

    # TODO



