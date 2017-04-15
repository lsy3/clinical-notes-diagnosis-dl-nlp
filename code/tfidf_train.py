import pandas as pd
import numpy as np
import cPickle
import argparse
import os, sys
from os.path import join
import tfidf_models
from keras.callbacks import ModelCheckpoint, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',      dest='nb_epoch', help='number of epoch', default=500, type = int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=128, type=int)
    parser.add_argument('--model_name', dest='model_name', help='model loaded from tfidf_model.py', default='nn_model_1', type=str)
    parser.add_argument('--pre_train', dest = 'pre_train', help='continue train from pretrained para? True/False', default=False)
    parser.add_argument('--data_file', dest='data_file', help='data file name', default='tfidf_v0_top10', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        print ('Use Default Settings ......')

    args = parser.parse_args()
    return args


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


def sparse2dense(data, feature_size):
    dense_data = np.zeros((len(data), feature_size))
    for i in range(len(data)):
        for j in data[i]:
            dense_data[i, j[0]] = j[1]
    return dense_data


def train(model_name, weights_path, train_data, train_label, valid_data, valid_label, feature_size, batch_size, nb_epoch, pre_train):
    function_list = tfidf_models.get_function_dict()
    model_func = function_list[model_name]
    model = model_func(feature_size)

    if pre_train == True:
        model.load_weights(weights_path)
    print ('checkpoint')
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    # model.fit_generator(generator=batch_generator(train_data, train_label, batch_size, True, feature_size),
    #                     nb_epoch=nb_epoch,
    #                     validation_data=batch_generator(valid_data, valid_label, batch_size, True, feature_size),
    #                     validation_steps=128,
    #                     samples_per_epoch= int(len(train_data) / batch_size),
    # 	            callbacks=[checkpointer, earlystopping])


    model.fit(train_data, train_label,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=[valid_data, valid_label],
              callbacks=[checkpointer, earlystopping])


def train_multi_label():
    args = parse_args()
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    model_name = args.model_name
    pre_train = args.pre_train
    data_file = args.data_file
    full_path = './data/' + data_file + '.p'
    f = open(full_path, 'rb')
    loaded_data = []
    for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        loaded_data.append(cPickle.load(f))
    f.close()

    train_data = loaded_data[0]
    valid_data = loaded_data[1]
    train_label = loaded_data[3]
    valid_label = loaded_data[4]
    feature_size = loaded_data[6]

    train_data = sparse2dense(train_data, feature_size)
    valid_data = sparse2dense(valid_data, feature_size)

    # train_data_mean = np.mean(train_data, axis = 0)
    # train_data_std = np.max(train_data, axis = 0)
    # f = open('./data/' + data_file + '_z.p', 'wb')
    # for obj in [train_data_mean, train_data_std]:
    #     cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    # f.close()
    # train_data -= train_data_mean
    # valid_data -= train_data_mean
    # train_data /= train_data_std
    # valid_data /= train_data_std
    if not os.path.isdir('./data/cache'):
        os.mkdir('./data/cache')
    weight_name = 'weight_' + model_name + '_' + data_file + '.h5'
    weights_path = join('./data/cache', weight_name)
    train(model_name, weights_path, train_data, train_label, valid_data, valid_label, feature_size, batch_size, nb_epoch,
          pre_train)


def train_multi_model():
    args = parse_args()
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    model_name = args.model_name
    pre_train = args.pre_train
    data_file = args.data_file
    full_path = './data/' + data_file + '.p'
    f = open(full_path, 'rb')
    loaded_data = []
    for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        loaded_data.append(cPickle.load(f))
    f.close()

    train_data = loaded_data[0]
    valid_data = loaded_data[1]
    feature_size = loaded_data[6]
    train_data = sparse2dense(train_data, feature_size)
    valid_data = sparse2dense(valid_data, feature_size)

    train_data_mean = np.mean(train_data, axis = 0)
    train_data_std = np.max(train_data, axis = 0)
    f = open('./data/' + data_file + '_z.p', 'wb')
    for obj in [train_data_mean, train_data_std]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    # train_data -= train_data_mean
    # valid_data -= train_data_mean
    # train_data /= train_data_std
    # valid_data /= train_data_std

    for i in range(10):
        train_label = loaded_data[3][:, i]
        valid_label = loaded_data[4][:, i]
        from keras.utils.np_utils import to_categorical
        train_label = to_categorical(train_label, num_classes=2)
        valid_label = to_categorical(valid_label, num_classes=2)
        if not os.path.isdir('./data/cache'):
            os.mkdir('./data/cache')
        weight_name = 'weight_' + model_name + '_' + data_file + '_' + str(i + 1) + '.h5'
        weights_path = join('./data/cache', weight_name)
        train(model_name, weights_path, train_data, train_label, valid_data, valid_label, feature_size, batch_size,
              nb_epoch, pre_train)


if __name__ == '__main__':
    train_multi_label()
