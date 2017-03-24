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

    embedding_size = embedding_matrix.shape[1]

    #LSTM with embedding trainable
    from keras.models import Model, Sequential
    from keras.layers import Input
    from keras.optimizers import Adam, RMSprop
    from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization
    from keras.layers import LSTM

    # opt = Adam(0.002)
    opt = RMSprop()
    inp = Input(shape=train_sequence.shape[1:])
    x = Embedding(len(dictionary) + 1,
                  embedding_size,
                  weights=[embedding_matrix],
                  input_length=max_sequence_length,
                  trainable=True,
                  input_shape=train_sequence.shape[1:])(inp)
    x = LSTM(256, return_sequences=False, dropout_W = 0.3, dropout_U = 0.3)(x)
    x = BatchNormalization()(x)

    category_number = train_label.shape[1]
    pred = Dense(category_number, activation='softmax')(x)

    model = Model(inp, pred)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    model.summary()
    model.fit(train_sequence, train_label, validation_split= 0.2, epochs=2, batch_size=64)

    predict_label = model.predict(test_sequence)
    predict_sum = np.sum(predict_label, axis=1)
    print(predict_sum[:10])
    scores = model.evaluate(test_sequence, test_label, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


