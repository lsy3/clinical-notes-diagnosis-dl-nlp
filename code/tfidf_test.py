import cPickle
import argparse
import sys
from os.path import join
import numpy as np
import dl_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=128, type=int)
    parser.add_argument('--model_name', dest='model_name', help='model loaded from dl_model.py', default='nn_model_1', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        print ('Run Default Settings ....... ')

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


def test(data_file):
    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    f = open(data_file, 'rb')
    loaded_data = []
    for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        loaded_data.append(cPickle.load(f))
    f.close()

    test_data = loaded_data[2]
    test_label = loaded_data[5][:,0]
    # from keras.utils.np_utils import to_categorical
    # test_label = to_categorical(test_label, num_classes=2)
    feature_size = loaded_data[6]

    file_path = './data/cache'
    weights_name = 'weights_' + model_name + '.h5'

    function_list = dl_models.get_function_dict()
    model_func = function_list[model_name]
    model = model_func(feature_size)
    model.load_weights(join(file_path, weights_name))
    print('Loaded model from disk')
    # convert sparse test data to dense
    test_data_dense = np.zeros((len(test_data), feature_size))
    for i in range(len(test_data)):
        for j in test_data[i]:
            test_data_dense[i, j[0]] = j[1]
    test_pred = model.predict_classes(test_data_dense, batch_size = batch_size, verbose=0)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_label,test_pred)
    print(cm)


if __name__ == '__main__':
    data_file = './data/tfidf_top10.p'
    test(data_file)