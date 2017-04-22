import cPickle
import argparse
import sys
import os
from os.path import join
import numpy as np
import wordseq_models
from sklearn.metrics import confusion_matrix
from keras.layers import Embedding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', dest='datafile', help='input pickle file', default='./data/DATA_WORDSEQV0_HADM_TOP10.p', type = str)
    parser.add_argument('--embmatrix', dest='embmatrix', help='embedding matrix', default='./data/EMBMATRIX_WORD2VEC_v2_100dim.p', type = str)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=128, type=int)
    parser.add_argument('--model_name', dest='model_name', help='model loaded from dl_model.py', default='conv1d_1', type=str)
    parser.add_argument('--append_name', dest='pre_train_append', help='load weights_model_name<append_name>', default='', type=str)
    parser.add_argument('--gpu', dest = 'gpu', help='set gpu no to be used (default: all)', default='',type=str)
    parser.add_argument('--patience', dest ='patience', help='patient for early stopper', default=5, type=int)
    parser.add_argument('--prob', dest ='prob', help='prob for activate the label', default=0.5, type=float)
    parser.add_argument('--argmax', dest='argmax', help='argmax trigger', default=False)
    parser.add_argument('--eval_topN', dest='eval_topN', help='evaluate only the top N labels (ordered by f1 score)', default=-1, type=int)
    parser.add_argument('--eval_firstN', dest='eval_firstN', help='evaluate only the first N labels', default=-1, type=int)

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

def evaluate(test_label, test_pred, gettopX=-1, getfirstX=-1):
    precision_list = np.zeros((test_label.shape[1]))
    recall_list = np.zeros((test_label.shape[1]))
    f1_list = np.zeros((test_label.shape[1]))
    accuracy_list = np.zeros((test_label.shape[1]))
    for i in range(test_label.shape[1]):
        cm = confusion_matrix(test_label[:, i],test_pred[:, i])
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
        print cm
    out = {'prec_mean': np.mean(precision_list),
           'prec_std': np.std(precision_list),
           'recall_mean': np.mean(recall_list),
           'recall_std': np.std(recall_list),
           'acc_mean': np.mean(accuracy_list),
           'acc_std': np.std(accuracy_list),
           'f1_mean': np.mean(f1_list),
           'f1_std': np.std(f1_list)}
    if np.isnan(np.sum(precision_list)):
        out['prec_mean2'] = np.mean(np.nan_to_num(precision_list))
        out['prec_std2'] = np.std(np.nan_to_num(precision_list))
    if np.isnan(np.sum(recall_list)):
        out['recall_mean2'] = np.mean(np.nan_to_num(recall_list))
        out['recall_std2'] = np.std(np.nan_to_num(recall_list))
    if np.isnan(np.sum(f1_list)):
        out['f1_mean2'] = np.mean(np.nan_to_num(f1_list))
        out['f1_std2'] = np.std(np.nan_to_num(f1_list))

    for code, num in [('top', gettopX), ('first', getfirstX)]:
        if num <= 0: continue

        if code == 'top':
            idx = np.argsort(np.nan_to_num(f1_list))[-num:]
        elif code == 'first':
            idx = xrange(num)

        out['prec_mean'+code] = np.mean(precision_list[idx])
        out['prec_std'+code] = np.std(precision_list[idx])
        out['recall_mean'+code] = np.mean(recall_list[idx])
        out['recall_std'+code] = np.std(recall_list[idx])
        out['acc_mean'+code] = np.mean(accuracy_list[idx])
        out['acc_std'+code] = np.std(accuracy_list[idx])
        out['f1_mean'+code] = np.mean(f1_list[idx])
        out['f1_std'+code] = np.std(f1_list[idx])
        if np.isnan(np.sum(precision_list[idx])):
            out['prec_mean'+code+'2'] = np.mean(np.nan_to_num(precision_list[idx]))
            out['prec_std'+code+'2'] = np.std(np.nan_to_num(precision_list[idx]))
        if np.isnan(np.sum(recall_list[idx])):
            out['recall_mean'+code+'2'] = np.mean(np.nan_to_num(recall_list[idx]))
            out['recall_std'+code+'2'] = np.std(np.nan_to_num(recall_list[idx]))
        if np.isnan(np.sum(f1_list[idx])):
            out['f1_mean'+code+'2'] = np.mean(np.nan_to_num(f1_list[idx]))
            out['f1_std'+code+'2'] = np.std(np.nan_to_num(f1_list[idx]))

    return out


def test(args):
    
    model_name = args.model_name
    batch_size = args.batch_size
    f = open(args.datafile, 'rb')
    loaded_data = []
    for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        loaded_data.append(cPickle.load(f))
    f.close()

    dictionary = loaded_data[0]
    train_sequence = loaded_data[1]
    #val_sequence = loaded_data[2]
    test_sequence = loaded_data[3]
    train_label = loaded_data[4]
    #val_label = loaded_data[5]
    test_label = loaded_data[6]

    f = open(args.embmatrix)
    embedding_matrix = cPickle.load(f)
    f.close

    max_sequence_length = test_sequence.shape[1]
    vocabulary_size = len(dictionary) + 1
    embedding_dim = embedding_matrix.shape[1]
    category_number = test_label.shape[1]
    input_shape = test_sequence.shape[1:]

    embedding_layer = Embedding(vocabulary_size,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_sequence_length,
                        trainable=False,
                        input_shape=input_shape)

    file_path = './data/cache'
    weights_name = 'weights_' + model_name + args.pre_train_append + '.h5'

    model_func = getattr(wordseq_models, model_name)
    model = model_func(input_shape, category_number, embedding_layer)
    model.load_weights(join(file_path, weights_name))
    print('Loaded model from disk')
    if args.argmax == True:
        test_pred = model.predict(test_sequence, batch_size = batch_size, verbose=0)
        test_pred[np.argmax(test_pred, axis=0)] = 1
        train_pred = model.predict(train_sequence, batch_size=batch_size, verbose=0)
        train_pred[np.argmax(train_pred, axis=0)] = 1
    else:
        test_pred = model.predict(test_sequence, batch_size = batch_size, verbose=0)
        test_pred[test_pred >= args.prob] = 1
        test_pred[test_pred < args.prob] = 0
        train_pred = model.predict(train_sequence, batch_size=batch_size, verbose=0)
        train_pred[train_pred >= args.prob] = 1
        train_pred[train_pred < args.prob] = 0

    trainEval = evaluate(train_label, train_pred, gettopX=args.eval_topN, getfirstX=args.eval_firstN)
    testEval = evaluate(test_label, test_pred, gettopX=args.eval_topN, getfirstX=args.eval_firstN)

    for code, num in [('', 1), ('top', args.eval_topN), ('first', args.eval_firstN)]:
        if num < 0: continue:

        print "{0}{1} {2}{3}".format(model_name, args.pre_train_append,
                                     code, num if code != '' else '')
        print "train: "
        for i in ['prec', 'recall', 'acc', 'f1']:
            print "{0}: {1} std: {2}".format(i,
                                         trainEval["{0}_mean{1}".format(i,code)],
                                         trainEval["{0}_std{1}".format(i,code)])
            if "{0}_mean{1}2".format(i,code) not in trainEval: continue
            print "{0}_zeronan: {1} std: {2}".format(i,
                                             trainEval["{0}_mean{1}2".format(i,code)],
                                             trainEval["{0}_std{1}2".format(i,code)])
    
        print "test:"
        for i in ['prec', 'recall', 'acc', 'f1']:
            print "{0}: {1} std: {2}".format(i,
                                         testEval["{0}_mean{1}".format(i,code)],
                                         testEval["{0}_std{1}".format(i,code)])
            if "{0}_mean{1}2".format(i,code) not in testEval: continue
            print "{0}_zeronan: {1} std: {2}".format(i,
                                             testEval["{0}_mean{1}2".format(i,code)],
                                             testEval["{0}_std{1}2".format(i,code)])

if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    test(args)
