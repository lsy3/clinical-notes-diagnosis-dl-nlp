from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, BatchNormalization, Activation
from keras.layers import LSTM



def nn_model_1(feature_size):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=2000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=2))
    model.add(Activation('sigmoid'))

    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def nn_model_2(feature_size):
    print('Build model...')
    model = Sequential()
    model.add(Dense(units=500, input_dim=feature_size))
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


def get_function_dict():
    funcs = {'nn_model_1': nn_model_1, 'nn_model_2': nn_model_2}
    return funcs

