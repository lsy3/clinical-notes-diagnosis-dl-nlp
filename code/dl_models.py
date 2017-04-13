from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization, LSTM

def nn_model_1(feature_size):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=2000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=2))
    model.add(Activation('sigmoid'))

    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_2(feature_size):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=1000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_3(feature_size):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=50, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=20))
    model.add(Activation('relu'))
    # model.add(Dense(units=500))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=100))
    # model.add(Activation('relu'))
    # model.add(Dense(units=100))
    # model.add(Activation('relu'))

    model.add(Dense(units=10))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_4(feature_size):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=1000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=2))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_5(feature_size):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=50, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=2))
    model.add(Activation('sigmoid'))

    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_6(feature_size):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=100, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=50))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=20))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(units=20))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dense(units=100))
    # model.add(Activation('relu'))

    model.add(Dense(units=10))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_7(feature_size):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=300, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

def lstm_model_1(vocabulary_size, embedding_matrix, max_sequence_length, category_number, input_shape):
    # LSTM with embedding trainable
    embedding_dim = embedding_matrix.shape[1]

    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocabulary_size,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_sequence_length,
                        trainable=True,
                        input_shape=input_shape))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(category_number, activation='sigmoid'))

    model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
    model.summary()


def get_function_dict():
    funcs = {'nn_model_1': nn_model_1,
             'nn_model_2': nn_model_2,
             'nn_model_3': nn_model_3,
             'nn_model_4': nn_model_4,
             'nn_model_5': nn_model_5,
             'nn_model_6': nn_model_6,
             'nn_model_7': nn_model_7,
             'lstm_model_1': lstm_model_1}
    return funcs

