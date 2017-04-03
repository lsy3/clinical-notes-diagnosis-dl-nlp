import pandas as pd
import numpy as np
import cPickle
import argparse
import os, sys
from os.path import join
import dl_models
from keras.callbacks import ModelCheckpoint, EarlyStopping



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',      dest='nb_epoch', help='number of epoch', default=50, type = int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=128, type=int)
    parser.add_argument('--model_name', dest='model_name', help='model loaded from dl_model.py', default='nn_model_1', type=str)
    parser.add_argument('--pre_train', dest = 'pre_train', help='continue train from pretrained para? True/False', default=False)
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


def train(data_file):
    args = parse_args()
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    model_name = args.model_name
    pre_train = args.pre_train

    f = open(data_file, 'rb')
    loaded_data = []
    for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        loaded_data.append(cPickle.load(f))
    f.close()

    train_data = loaded_data[0]
    valid_data = loaded_data[1]
    # train_label = loaded_data[3]
    train_label = loaded_data[3][:, 0] # test on only the first icd9code
    valid_label = loaded_data[4][:, 0]
    from keras.utils.np_utils import to_categorical
    train_label = to_categorical(train_label, num_classes=2)
    valid_label = to_categorical(valid_label, num_classes=2)


    feature_size = loaded_data[6]

    function_list = dl_models.get_function_dict()
    model_func = function_list[model_name]
    model = model_func(feature_size)

    if not os.path.isdir('./data/cache'):
        os.mkdir('./data/cache')
    weight_name = 'weights_' + model_name + '.h5'
    weights_path = join('./data/cache', weight_name)
    if pre_train == True:
        model.load_weights(weights_path)
    print ('checkpoint')
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    model.fit_generator(generator=batch_generator(train_data, train_label, batch_size, True, feature_size),
                        nb_epoch=nb_epoch,
                        validation_data=batch_generator(valid_data, valid_label, batch_size, True, feature_size),
                        validation_steps=128,
                        samples_per_epoch= int(len(train_data) / batch_size),
			            callbacks=[checkpointer, earlystopping])


if __name__ == '__main__':
    data_file = './data/tfidf_top10.p'
    train(data_file)
