import cPickle
import argparse
import sys
import os
import numpy as np
import wordseq_models
from evaluate import *
from keras.layers import Embedding
from os import listdir
from os.path import isfile, join
import pandas as pd

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
    parser.add_argument('--labelmode', dest ='labelmode', 
                        help='additional label processing. Option: tile<num>, repeat<num>',
                        default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        print ('Run Default Settings ....... ')

    args = parser.parse_args()
    return args

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
    # val_sequence = loaded_data[2]
    test_sequence = loaded_data[3]
    train_label = loaded_data[4]
    # val_label = loaded_data[5]
    test_label = loaded_data[6]

    if args.labelmode[:4] == 'tile':
        print 'labelmode: tile'
        n = int(args.labelmode[4:].strip()
        train_label = np.tile(train_label, n)
        test_label = np.tile(test_label, n)
    elif args.labelmode[:6] == 'repeat':
        print 'labelmode: repeat'
        n = int(args.labelmode[6:].strip()
        train_label = np.repeat(train_label, n, axis=1)
        test_label = np.repeat(test_label, n, axis=1)

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

    trainEval = evaluate_1(train_label, train_pred, gettopX=args.eval_topN, getfirstX=args.eval_firstN)
    testEval = evaluate_1(test_label, test_pred, gettopX=args.eval_topN, getfirstX=args.eval_firstN)

    for code, num in [('', 1), ('top', args.eval_topN), ('first', args.eval_firstN)]:
        if num < 0: continue

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
        print ""


def test_auto(args):
    file_path = './data/auto_test_holder/models'
    model_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    for mf in model_files:
        weights_name = join(file_path, mf)
        parts = mf.split('_')

        # parts[3]  version
        # parts[4]  code cat
        # parts[5]  feature number
        if parts[5][0] == 'f':
            embedding_name = 'EMBMATRIXV' + parts[3][1] + '_WORD2VEC_v2_' + parts[5][1:4] + 'dim.p'
        if parts[5][0] == 'b':
            embedding_name = 'EMBMATRIXV' + parts[3][1] + '_BIONLPVEC_Pubmedshufflewin' + parts[5][-5:-3] + 'F.p'
        embedding_path = './data/auto_test_holder/embedding_matrix'
        f = open(join(embedding_path, embedding_name))
        embedding_matrix = cPickle.load(f)
        f.close

        model_name = parts[1] + '_' + parts[2]

        category_dict = {'cat': 'TOP10CAT', 'cod': 'TOP10', 'c5t': 'TOP50CAT', 'c5d': 'TOP50'}
        data_file = './data/DATA_WORDSEQV' + parts[3][1] + '_HADM_' + category_dict[parts[4][:3]] + '.p'
        f = open(data_file, 'rb')
        loaded_data = []
        for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
            loaded_data.append(cPickle.load(f))
        f.close()

        dictionary = loaded_data[0]
        train_sequence = loaded_data[1]
        # val_sequence = loaded_data[2]
        test_sequence = loaded_data[3]
        train_label = loaded_data[4]
        # val_label = loaded_data[5]
        test_label = loaded_data[6]


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

     
        model_func = getattr(wordseq_models, model_name)
        model = model_func(input_shape, category_number, embedding_layer)
	print weights_name
        model.load_weights(weights_name)
        print('Loaded model from disk')

        test_pred = model.predict(test_sequence, batch_size=128, verbose=0)
        train_pred = model.predict(train_sequence, batch_size=128, verbose=0)

        trainEval = evaluate_4(train_label, train_pred)
        testEval = evaluate_4(test_label, test_pred)

        df = pd.DataFrame(trainEval)
        df.to_csv(data_file[:-2] + '_test_res.csv')
        df = pd.DataFrame(testEval)
        df.to_csv(data_file[:-2] + '_train_res.csv')


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    #test_auto(args)
    test(args)
