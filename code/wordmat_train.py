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
    parser.add_argument('--model_name', dest='model_name', help='model loaded from *_model.py', default='conv2d_1', type=str)
    parser.add_argument('--append_name', dest='append_name', help='load weights_model_name<append_name>', default='', type=str)
    parser.add_argument('--pre_train', dest = 'pre_train', help='continue train from pretrained para? True/False', default=False)
    parser.add_argument('--gpu', dest = 'gpu', help='set gpu no to be used (default: 0)', default='7',type=str)
    parser.add_argument('--plot_model', dest = 'plot_model', help='plot the said model', default=True)
    parser.add_argument('--patience', dest ='patience', help='patient for early stopper', default=5, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        print ('Use Default Settings ......')

    args = parser.parse_args()
    return args
	
	
def get_embedded_data(sequence, embedding_matrix):
    embedded_data = np.zeros((len(sequence), embedding_matrix.shape[1]))
    for i in range(len(sequence)):
        embedded_data[i, :] = embedding_matrix[sequence[i], :]
	return embedded_data
		

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

    f = open(args.embmatrix)
    embedding_matrix = cPickle.load(f)
    f.close()

    max_sequence_length = train_sequence.shape[1]
    vocabulary_size = len(dictionary) + 1
    embedding_dim = embedding_matrix.shape[1]
    category_number = train_label.shape[1]

    train_matrix = np.array(map(lambda x: get_embedded_data(x, embedding_matrix), train_sequence))
    val_matrix = np.array(map(lambda x: get_embedded_data(x, embedding_matrix), val_sequence))
    # train_matrix = train_matrix.reshape(train_matrix.shape[0], 1, train_matrix.shape[1], train_matrix.shape[2])
    print train_matrix.shape

    # from keras.layers import Embedding
    # from keras.models import Sequential
    # input_shape = train_sequence.shape[1:]
    # embedding_layer = Embedding(vocabulary_size,
    #                     embedding_dim,
    #                     weights=[embedding_matrix],
    #                     input_length=max_sequence_length,
    #                     trainable=False,
    #                     input_shape=input_shape)
    # #
    # model = Sequential()
    # model.add(embedding_layer)
    #
    # from keras import backend as K
    # get_3rd_layer_output = K.function([model.layers[0].input],
    #                                   [model.layers[3].output])
    # layer_output = get_3rd_layer_output([X])[0]


    # full_name = './data/train_matrix_2d.h5'
    # import h5py
    # hf = h5py.File(full_name, 'r')
    # train_matrix = np.copy(hf['train_matrix'])
    # hf.close()

    # with h5py.File(full_name, 'w') as hf:
    #     hf.create_dataset('train_matrix', data=train_matrix)
    # print "done"
    from keras import backend as K
    row = train_matrix.shape[1]
    col = train_matrix.shape[2]
    if K.image_data_format() == 'channels_first':
        train_matrix = train_matrix.reshape(train_matrix.shape[0], 1, row, col)
        print train_matrix.shape
        input_shape = (1, row, col)
    else:
        train_matrix = train_matrix.reshape(train_matrix.shape[0], row, col, 1)
        print train_matrix.shape
        input_shape = (row, col, 1)
	
    model_func = getattr(wordseq_models, model_name)
    model = model_func(input_shape, category_number)

    if args.plot_model:
        fig_name = './data/cache/' + args.model_name + '.png'
        plot_model(model, fig_name, True, False)


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

    model.fit(train_matrix, train_label,
              batch_size = batch_size,
              epochs = nb_epoch,
              validation_data = [val_matrix, val_label],
              callbacks=[checkpointer, earlystopping])


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train(args)
