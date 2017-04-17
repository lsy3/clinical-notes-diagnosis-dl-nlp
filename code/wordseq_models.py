from keras.models import *
from keras.layers import *

def lstm_model_1(input_shape, output_shape, embedding_layer):
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
    model.summary()
    return model


def lstm_model_2(input_shape, output_shape, embedding_layer):
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
    model.summary()
    return model


def lstm_model_3(input_shape, output_shape, embedding_layer):
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
    model.summary()
    return model

	
def lstm_model_4(input_shape, output_shape, embedding_layer):
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
    model.summary()
    return model
	

def lstm_model_5(input_shape, output_shape, embedding_layer):
	print('Build model...')
	model = Sequential()
	model.add(embedding_layer)
	model.add(LSTM(128, return_sequences=True))
	model.add(LSTM(64, return_sequences=True))
	model.add(LSTM(32))
	model.add(Dense(output_shape, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
	model.summary()
	return model
	
	
def lstm_model_6(input_shape, output_shape, embedding_layer):
	print('Build model...')
	model = Sequential()
	model.add(embedding_layer)
	model.add(LSTM(256, return_sequences=True))
	model.add(LSTM(128, return_sequences=True))
	model.add(LSTM(64))
	model.add(Dense(output_shape, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
	model.summary()
	return model
	

def lstm_model_7(input_shape, output_shape, embedding_layer):
	model = Sequential()
	model.add(embedding_layer)
	model.add(Conv1D(256, 5, padding='valid', activation='relu', strides=1))
	model.add(MaxPooling1D(4))
	model.add(Conv1D(256, 5, padding='valid', activation='relu', strides=1))
	model.add(MaxPooling1D(4))
	model.add(LSTM(256))
	model.add(Dense(output_shape, activation='sigmoid'))
	
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
	model.summary()
	return model


def lstm_model_8(input_shape, output_shape, embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(GRU(256)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model


def lstm_model_9(input_shape, output_shape, embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model

	
	
def rnn_model_1(input_shape, output_shape, embedding_layer):
	print('Build model...')
	model = Sequential()
	model.add(embedding_layer)
	model.add(SimpleRNN(128, return_sequences=True))
	model.add(SimpleRNN(128, return_sequences=True))
	model.add(SimpleRNN(128))
	model.add(Dense(output_shape, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
	model.summary()
	return model
	

def lstm_model_glove_1(input_shape, output_shape, embedding_layer):
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
    model.summary()
    return model


def conv1d_1(input_shape, output_shape, embedding_layer):
    sequence_input = Input(shape=input_shape, dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', padding='same')(embedded_sequences)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(35, padding='same')(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(output_shape, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model

def conv1d_2(input_shape, output_shape, embedding_layer):
    sequence_input = Input(shape=input_shape, dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', padding='valid')(embedded_sequences)
    x = MaxPooling1D(5, padding='valid')(x)
    x = Conv1D(128, 5, activation='relu', padding='valid')(x)
    x = MaxPooling1D(5, padding='valid')(x)
    x = Conv1D(128, 5, activation='relu', padding='valid')(x)
    x = MaxPooling1D(35, padding='valid')(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(output_shape, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model


def conv1d_3(input_shape, output_shape, embedding_layer):
    sequence_input = Input(shape=input_shape, dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(256, 5, activation='relu', padding='same')(embedded_sequences)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(256, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(256, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(35, padding='same')(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    preds = Dense(output_shape, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model


def conv1d_8(input_shape, output_shape, embedding_layer):
    sequence_input = Input(shape=input_shape, dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', padding='same')(embedded_sequences)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(35, padding='same')(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(output_shape, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model


def conv2d_1(input_shape, output_shape):

    print('Build model...')
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
    model.summary()
    return model
	
	
def conv2d_2(input_shape, output_shape):
    print('Build model...')
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=input_shape,
                   padding='same', return_sequences=True))
    model.add(BatchNormalization())
	
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
    model.add(BatchNormalization())


    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
	
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
    model.summary()
    return model


def gru_1(input_shape, output_shape, embedding_layer):
    model = Sequential()

    #model.add(Input(shape=input_shape, dtype='int32'))
    model.add(embedding_layer)
    model.add(GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))  # returns a sequence of vectors of dimension 32
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))  # return a single vector of dimension 32
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model


def gru_2(input_shape, output_shape, embedding_layer):
    model = Sequential()

    #model.add(Input(shape=input_shape, dtype='int32'))
    model.add(embedding_layer)
    model.add(GRU(256))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model


def gru_3(input_shape, output_shape, embedding_layer):
    model = Sequential()

    #model.add(Input(shape=input_shape, dtype='int32'))
    model.add(embedding_layer)
    model.add(GRU(128, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(Dense(output_shape/10, activation='relu'))
    model.add(Flatten())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model


def gru_4(input_shape, output_shape, embedding_layer):
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(GRU(64))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model


def gru_8(input_shape, output_shape, embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(256)))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
    model.summary()
    return model

