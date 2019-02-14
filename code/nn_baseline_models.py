from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization, LSTM


def nn_model_1(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=1000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))

    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_2(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=1000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_3(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=1000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=1000))
    model.add(Activation('relu'))

    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_4(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=500, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=50))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_5(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=300, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_6(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=500, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=300))
    model.add(Activation('relu'))
    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_7(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=1000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

def nn_model_8(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=10000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))

    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

def nn_model_9(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=5000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))

    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

def nn_model_10(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=10000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=1000))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

def nn_model_11(feature_size, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=5000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=output_shape))
    model.add(Activation('sigmoid'))
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


