import pandas as pd
import numpy as np
import cPickle

def parse(s):
    """
    Parse string representation back into the SparseVector.
    """
    start = s.find('(')
    if start == -1:
        raise ValueError("Tuple should start with '('")
    end = s.find(')')
    if end == -1:
        raise ValueError("Tuple should end with ')'")
    s = s[start + 1: end].strip()

    size = s[: s.find(',')]
    try:
        size = int(size)
    except ValueError:
        raise ValueError("Cannot parse size %s." % size)

    ind_start = s.find('[')
    if ind_start == -1:
        raise ValueError("Indices array should start with '['.")
    ind_end = s.find(']')
    if ind_end == -1:
        raise ValueError("Indices array should end with ']'")
    new_s = s[ind_start + 1: ind_end]
    ind_list = new_s.split(',')
    try:
        indices = [int(ind) for ind in ind_list if ind]
    except ValueError:
        raise ValueError("Unable to parse indices from %s." % new_s)
    s = s[ind_end + 1:].strip()

    val_start = s.find('[')
    if val_start == -1:
        raise ValueError("Values array should start with '['.")
    val_end = s.find(']')
    if val_end == -1:
        raise ValueError("Values array should end with ']'.")
    val_list = s[val_start + 1: val_end].split(',')
    try:
        values = [float(val) for val in val_list if val]
    except ValueError:
        raise ValueError("Unable to parse values from %s." % s)
    return size, zip(indices, values)


def csv2pickle():
    label_col = [i + 1 for i in range(10)]
    df_label = pd.read_csv('./data/DATA_TFIDF_HADM_TOP10.csv', usecols=label_col)
    label = df_label.values

    df_features = pd.read_csv('./data/DATA_TFIDF_HADM_TOP10.csv', usecols=[11])
    features = df_features.values
    size, _ = parse(features[0][0])
    sparse_list = []

    for i in range(len(features)):
        if i % 10000 == 0:
            print i
        _, val = parse(features[i][0])
        sparse_list.append(val)

    del features

    f = open('./data/tfidf_top10.p', 'wb')
    for obj in [sparse_list, label, size]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


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


def train():
    f = open('./data/tfidf_top10.p', 'rb')
    loaded_data = []
    for i in range(3):  # [features, label, feature_size]:
        loaded_data.append(cPickle.load(f))
    f.close()

    features = loaded_data[0]
    label = loaded_data[1]
    feature_size = loaded_data[2]
    # split train and test
    train_percentage = 0.7
    test_percentage = 0.15
    indices = np.random.permutation(len(features))
    train_valid_split = int(train_percentage * len(features))
    valid_test_split = int(test_percentage * len(features))

    train_idx = indices[:train_valid_split]
    valid_idx = indices[train_valid_split:-valid_test_split]
    test_idx = indices[:-valid_test_split]


    train_data = list(features[i] for i in train_idx)
    valid_data = list(features[i] for i in valid_idx)
    test_data = list(features[i] for i in test_idx)

    train_label, valid_label, test_label = label[train_idx], label[valid_idx], label[test_idx]

    # LSTM with embedding trainable
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Embedding, BatchNormalization, Activation
    from keras.layers import LSTM


    print('Build model...')
    model = Sequential()
    model.add(Dense(units=1000, input_dim=feature_size))
    model.add(Activation('relu'))
    model.add(Dense(units=1000))
    model.add(Activation('relu'))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['mse'])
    model.summary()
    model.fit_generator(generator=batch_generator(train_data, train_label, 512, True, feature_size),
                        nb_epoch=10,
                        validation_data=batch_generator(valid_data, valid_label, 512, True, feature_size),
                        validation_steps=128,
                        samples_per_epoch=30000)

    # serialize model to JSON
    model_json = model.to_json()
    with open("./data/tfidf_top10_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./data/tfidf_top10_model.h5")
    print("Saved model to disk")

    t = model.evaluate_generator(generator=batch_generator(test_data, test_label, 32, True, feature_size))
    print (t)

def test():
    pass



if __name__ == '__main__':
    train()
