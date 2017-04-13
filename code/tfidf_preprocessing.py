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

    train_percentage = 0.7
    test_percentage = 0.15
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

    print train_data
    print valid_data
    print test_data
    print train_label
    print valid_label
    print test_label
    print size
    f = open('./data/tfidf_top10.p', 'wb')
    for obj in [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


if __name__ == '__main__':
    csv2pickle()

