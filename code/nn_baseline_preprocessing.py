import pandas as pd
import numpy as np
import cPickle
import h5py

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


def csv2pickle(file_name):
    label_col = [i + 1 for i in range(10)]
    df_label = pd.read_csv(file_name, usecols=label_col)
    label = df_label.values

    df_features = pd.read_csv(file_name, usecols=[11])
    features = df_features.values
    size, _ = parse(features[0][0])
    sparse_list = []

    for i in range(len(features)):
        if i % 10000 == 0:
            print i
        _, val = parse(features[i][0])
        sparse_list.append(val)

    del features

    train_percentage = 0.5
    test_percentage = 0.25
    np.random.seed(seed=42)
    indices = np.random.permutation(len(sparse_list))
    train_valid_split = int(train_percentage * len(sparse_list))
    valid_test_split = int(test_percentage * len(sparse_list))

    train_idx = indices[:train_valid_split]
    valid_idx = indices[train_valid_split:-valid_test_split]
    test_idx = indices[:-valid_test_split]

    train_data = list(sparse_list[i] for i in train_idx)
    valid_data = list(sparse_list[i] for i in valid_idx)
    test_data = list(sparse_list[i] for i in test_idx)

    train_label, valid_label, test_label = label[train_idx], label[valid_idx], label[test_idx]

    f = open('./data/tfidf_top10.p', 'wb')
    for obj in [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def csv2sparse(file_name, label_number):
    label_col = [i + 1 for i in range(label_number)]
    df_label = pd.read_csv(file_name, usecols=label_col)
    label = df_label.values

    df_features = pd.read_csv(file_name, usecols=[label_number + 1])
    features = df_features.values
    size, _ = parse(features[0][0])
    sparse_list = []

    for i in range(len(features)):
        if i % 10000 == 0:
            print i
        _, val = parse(features[i][0])
        sparse_list.append(val)

    return sparse_list, label, size


if __name__ == '__main__':
    # train_data, train_label, size = csv2sparse('./data/DATA_TFIDFV0_HADM_TOP50CAT_train.csv')
    # valid_data, valid_label, size = csv2sparse('./data/DATA_TFIDFV0_HADM_TOP50CAT_val.csv')
    # test_data, test_label, size = csv2sparse('./data/DATA_TFIDFV0_HADM_TOP50CAT_test.csv')

    feature_list = ['TFIDFV0', 'TFIDFV1', 'WORD2VECV0', 'WORD2VECV1', 'WORD2VECV2', 'WORD2VECV3', 'WORD2VECV4', 'DOC2VECV0', 'DOC2VECV1', 'DOC2VECV2']
    type_list = ['10', '10CAT', '50', '50CAT']
    #type_list = ['10']
    for feature in feature_list:
        for tp in type_list:
            train_file_name = './data/DATA_' + feature + '_HADM_TOP' + tp + '_train.csv'
            valid_file_name = './data/DATA_' + feature + '_HADM_TOP' + tp + '_val.csv'
            test_file_name = './data/DATA_' + feature + '_HADM_TOP' + tp + '_test.csv'
            print feature, tp

            train_data, train_label, size = csv2sparse(train_file_name, int(tp[:2]))
            valid_data, valid_label, size = csv2sparse(valid_file_name, int(tp[:2]))
            test_data, test_label, size = csv2sparse(test_file_name, int(tp[:2]))

    # train_data, train_label, size = csv2sparse('./data/DATA_WORD2VEC_HADM_TOP10_train.csv')
    # valid_data, valid_label, size = csv2sparse('./data/DATA_WORD2VEC_HADM_TOP10_val.csv')
    # test_data, test_label, size = csv2sparse('./data/DATA_WORD2VEC_HADM_TOP10_test.csv')

            f = open('./data/' + feature + '_' + tp + '.p', 'wb')
            for obj in [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
                cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()


