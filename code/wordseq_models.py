from keras.models import *
from keras.layers import *

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

def gru_1(input_shape, output_shape, embedding_layer):
    model = Sequential()

    #model.add(Input(shape=input_shape, dtype='int32'))
    model.add(embedding_layer)
    model.add(GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))  # returns a sequence of vectors of dimension 32
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))  # return a single vector of dimension 32
    model.add(Dense(output_shape, activation='softmax'))

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
