import pandas as pd
import numpy as np
import cPickle
import argparse
import os, sys
# import tensorflow as tf
from os.path import join
import wordseq_models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
from keras.utils import plot_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile',  dest='datafile', help='input pickle file', default='./data/DATA_WORDSEQV0_HADM_TOP10.p', type = str)
    parser.add_argument('--embmatrix', dest='embmatrix', help='embedding matrix', default='./data/EMBMATRIXV0_WORD2VEC_v2_300dim.p', type = str)
    parser.add_argument('--epoch',      dest='nb_epoch', help='number of epoch', default=50, type = int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=128, type = int)
    parser.add_argument('--model_name', dest='model_name', help='model loaded from *_model.py', default='conv1d_1', type=str)
    parser.add_argument('--append_name', dest='append_name', help='load weights_model_name<append_name>', default='', type=str)
    parser.add_argument('--pre_train', dest = 'pre_train', help='continue train from pretrained para? True/False', default=False)
    parser.add_argument('--gpu', dest = 'gpu', help='set gpu no to be used (default: 0)', default='1',type=str)
    parser.add_argument('--plot_model', dest = 'plot_model', help='plot the said model', default=True)
    parser.add_argument('--patience', dest ='patience', help='patient for early stopper', default=5, type=int)
    parser.add_argument('--labelmode', dest ='labelmode', 
                        help='additional label processing. Option: tile<num>, repeat<num>',
                        default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        print ('Use Default Settings ......')

    args = parser.parse_args()
    return args


def train(args):
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    model_name = args.model_name
    pre_train = args.pre_train

    f = open(args.datafile, 'rb')
    loaded_data = []
    for i in range(7): # [reverse_dictionary, train_sequence, test_sequence, train_label, test_label]:
        loaded_data.append(cPickle.load(f))
    f.close()

    dictionary = loaded_data[0]
    train_sequence = loaded_data[1]
    val_sequence = loaded_data[2]
    train_label = loaded_data[4]
    val_label = loaded_data[5]

    if args.labelmode[:4] == 'tile':
        n = int(args.labelmode[4:].strip())
        train_label = np.tile(train_label, n)
        val_label = np.tile(val_label, n)
        print 'labelmode: tile {0}'.format(train_label.shape)
    elif args.labelmode[:6] == 'repeat':
        n = int(args.labelmode[6:].strip())
        train_label = np.repeat(train_label, n, axis=1)
        val_label = np.repeat(val_label, n, axis=1)
        print 'labelmode: repeat {0}'.format(train_label.shape)
    elif args.labelmode[:5] == 'range':
        n = [int(i) for i in args.labelmode[5:].split(" ")]
        train_label = train_label[:,n[0]:n[1]]
        val_label = val_label[:,n[0]:n[1]]
        print 'labelmode: repeat {0}'.format(train_label.shape)

    # f = open('./data/dictionary_v0.p', 'wb')
    # cPickle.dump(dictionary, f)
    # f.close()

    f = open(args.embmatrix)
    embedding_matrix = cPickle.load(f)
    f.close()

    max_sequence_length = train_sequence.shape[1]
    vocabulary_size = len(dictionary) + 1
    embedding_dim = embedding_matrix.shape[1]
    category_number = train_label.shape[1]
    input_shape = train_sequence.shape[1:]

    embedding_layer = Embedding(vocabulary_size,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_sequence_length,
                        trainable=False,
                        input_shape=input_shape)

    model_func = getattr(wordseq_models, model_name)
    model = model_func(input_shape, category_number, embedding_layer)

    if args.plot_model:
        fig_name = './data/cache/' + args.model_name + '.png'
        plot_model(model, fig_name, True, False)
        # return

    if not os.path.isdir('./data/cache'):
        os.mkdir('./data/cache')
    weight_name = 'weights_' + model_name + args.append_name + '.h5'
    weights_path = join('./data/cache', weight_name)
    if pre_train:
        model.load_weights(weights_path)

    print ('checkpoint')
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=0, mode='auto')
    print('early stop at ', args.patience)

    #train_sequence = np.concatenate((train_sequence, val_sequence), axis=0)
    #train_label = np.concatenate((train_label, val_label), axis=0)

    model.fit(train_sequence, train_label,
              batch_size = batch_size,
              epochs = nb_epoch,
              validation_data = [val_sequence, val_label],
              callbacks=[checkpointer, earlystopping])


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train(args)
