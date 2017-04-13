import cPickle
import argparse
import sys
from os.path import join
import numpy as np
import dl_models
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=128, type=int)
    parser.add_argument('--model_name', dest='model_name', help='model loaded from dl_model.py', default='nn_model_2', type=str)
    parser.add_argument('--data_file', dest='data_file', help='data file name', default='tfidf_v0_top10', type=str)
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


def test_multi_label():
    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    data_file = args.data_file
    full_path = './data/' + data_file + '.p'
    f = open(full_path, 'rb')
    loaded_data = []
    for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        loaded_data.append(cPickle.load(f))
    f.close()

    test_data = loaded_data[0]
    test_label = loaded_data[3]
    # test_data = loaded_data[2]
    # test_label = loaded_data[5]
    # from keras.utils.np_utils import to_categorical
    # test_label = to_categorical(test_label, num_classes=2)
    feature_size = loaded_data[6]

    file_path = './data/cache'
    weights_name = 'weight_' + model_name + '_' + data_file + '.h5'

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

    # full_path = './data/tfidf_word2vec_top10_z.p'
    # f = open(full_path, 'rb')
    # loaded_data = []
    # for i in range(2):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
    #     loaded_data.append(cPickle.load(f))
    # f.close()
    # test_data_dense -= loaded_data[0]
    # test_data_dense /= loaded_data[1]
    test_pred = model.predict(test_data_dense, batch_size = batch_size, verbose=0)
    test_pred[test_pred >= 0.5] = 1
    test_pred[test_pred < 0.5] = 0

    precision_list = np.zeros((test_label.shape[1]))
    recall_list = np.zeros((test_label.shape[1]))
    f1_list = np.zeros((test_label.shape[1]))
    accuracy_list = np.zeros((test_label.shape[1]))

    for i in range(test_label.shape[1]):
        cm = confusion_matrix(test_label[:, i],test_pred[:, i])
        print cm
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        # tn | fp
        # ---|---
        # fn | tp
        precision = tp / float(tp + fp)
        precision_list[i] = precision
        recall = tp / float(tp + fn)
        recall_list[i] = recall
        f1 = 2 * (precision * recall / float(precision + recall))
        f1_list[i] = f1
        accuracy = (tp + tn) / float(tp + tn + fp + fn)
        accuracy_list[i] = accuracy

    print "precision: ", np.mean(precision_list), "std: ", np.std(precision_list)
    print "recall: ", np.mean(recall_list), "std: ", np.std(recall_list)
    print "accuracy: ", np.mean(accuracy_list), "std: ", np.std(accuracy_list)
    print "f1: ", np.mean(f1_list), "std: ", np.std(f1_list)
    print "precision_list: ", precision_list
    print "recall_list: ", recall_list
    print "accuracy_list: ", accuracy_list
    print "f1_list: ", f1_list


def test_multi_model():
    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    data_file = args.data_file
    full_path = './data/' + data_file + '.p'
    f = open(full_path, 'rb')
    loaded_data = []
    for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        loaded_data.append(cPickle.load(f))
    f.close()

    test_data = loaded_data[2]
    feature_size = loaded_data[6]
    test_label = loaded_data[5]
    precision_list = np.zeros((test_label.shape[1]))
    recall_list = np.zeros((test_label.shape[1]))
    f1_list = np.zeros((test_label.shape[1]))
    accuracy_list = np.zeros((test_label.shape[1]))

    for i in range(test_label.shape[1]):
        test_single = test_label[:, i]

        file_path = './data/cache'
        weights_name = 'weight_' + model_name + '_' + data_file + '_' + str(i+1) + '.h5'

        function_list = dl_models.get_function_dict()
        model_func = function_list[model_name]
        model = model_func(feature_size)
        model.load_weights(join(file_path, weights_name))
        print('Loaded model from disk')
        # convert sparse test data to dense
        test_data_dense = np.zeros((len(test_data), feature_size))
        for ii in range(len(test_data)):
            for j in test_data[ii]:
                test_data_dense[ii, j[0]] = j[1]
        test_pred = model.predict_classes(test_data_dense, batch_size = batch_size, verbose=0)
        # test_pred[test_pred >= 0.5] = 1
        # test_pred[test_pred < 0.5] = 0
        cm = confusion_matrix(test_single, test_pred)
        print cm
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        # tn | fp
        # ---|---
        # fn | tp
        precision = tp / float(tp + fp)
        precision_list[i] = precision
        recall = tp / float(tp + fn)
        recall_list[i] = recall
        f1 = 2 * (precision * recall / float(precision + recall))
        f1_list[i] = f1
        accuracy = (tp + tn) / float(tp + tn + fp + fn)
        accuracy_list[i] = accuracy

    print "precision: ", np.mean(precision_list), "std: ", np.std(precision_list)
    print "recall: ", np.mean(recall_list), "std: ", np.std(recall_list)
    print "accuracy: ", np.mean(accuracy_list), "std: ", np.std(accuracy_list)
    print "f1: ", np.mean(f1_list), "std: ", np.std(f1_list)


if __name__ == '__main__':
    test_multi_label()