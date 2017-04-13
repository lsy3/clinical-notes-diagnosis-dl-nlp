from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import cPickle
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.metrics import average_precision_score

## Keras sample code
# X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
# y2 = shuffle(y1, random_state=1)
# y3 = shuffle(y1, random_state=2)
# Y = np.vstack((y1, y2, y3)).T
# n_samples, n_features = X.shape # 10,100
# n_outputs = Y.shape[1] # 3
# n_classes = 3
# forest = RandomForestClassifier(n_estimators=100, random_state=1)
# multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
# multi_target_forest.fit(X, Y).predict(X)

def sparse2dense(data, feature_size):
    dense_data = np.zeros((len(data), feature_size))
    for i in range(len(data)):
        for j in data[i]:
            dense_data[i, j[0]] = j[1]
    return dense_data

data_file = './data/tfidf_v0_top10.p'
f = open(data_file, 'rb')
loaded_data = []
for i in range(7):  # [train_data, valid_data, test_data, train_label, valid_label, test_label, size]:
    loaded_data.append(cPickle.load(f))
f.close()

train_data = loaded_data[0]
valid_data = loaded_data[1]
test_data = loaded_data[2]
feature_size = loaded_data[6]
train_label = loaded_data[3] # test on only the first icd9code
valid_label = loaded_data[4]
test_label = loaded_data[5]


# convert sparse data to dense
train_data = sparse2dense(train_data, feature_size)
valid_data = sparse2dense(valid_data, feature_size)
test_data = sparse2dense(test_data, feature_size)

precision_list = np.zeros((10))
recall_list = np.zeros((10))
f1_list = np.zeros((10))
accuracy_list = np.zeros((10))

precision_list_2 = np.zeros((10))
recall_list_2 = np.zeros((10))
f1_list_2 = np.zeros((10))

for i in range(10):
    train_label = loaded_data[3][:, i]
    test_label = loaded_data[5][:, i]
    clf = RandomForestClassifier(max_depth=10, n_estimators=100)
    clf.fit(train_data, train_label)
    clf_probs = clf.predict_proba(test_data)
    clf_class = clf.predict(test_data)
    score = log_loss(test_label, clf_probs)
    # test_pred[test_pred >= 0.5] = 1
    # test_pred[test_pred < 0.5] = 0
    cm = confusion_matrix(test_label, clf_class)
    print cm
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    precision = precision_score(test_label, clf_class, average='micro')
    recall = recall_score(test_label, clf_class, average='macro')
    f1 = f1_score(test_label, clf_class)
    precision_list_2[i] = precision
    recall_list_2[i] = recall
    f1_list_2[i] = f1

    print precision
    print recall
    print f1

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




# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(50, 10), random_state=1)
# clf.fit(train_data, train_label)
# test_pred = clf.predict(test_data)
#
# precision_list = np.zeros((test_label.shape[1]))
# recall_list = np.zeros((test_label.shape[1]))
# f1_list = np.zeros((test_label.shape[1]))
# accuracy_list = np.zeros((test_label.shape[1]))
# for i in range(test_label.shape[1]):
#     cm = confusion_matrix(test_label[:, i], test_pred[:, i])
#     tn = cm[0, 0]
#     fp = cm[0, 1]
#     fn = cm[1, 0]
#     tp = cm[1, 1]
#     # tn | fp
#     # ---|---
#     # fn | tp
#     precision = tp / float(tp + fp)
#     precision_list[i] = precision
#     recall = tp / float(tp + fn)
#     recall_list[i] = recall
#     f1 = 2 * (precision * recall / float(precision + recall))
#     f1_list[i] = f1
#     accuracy = (tp + tn) / float(tp + tn + fp + fn)
#     accuracy_list[i] = accuracy
#
# print "precision: ", np.mean(precision_list), "std: ", np.std(precision_list)
# print "recall: ", np.mean(recall_list), "std: ", np.std(recall_list)
# print "accuracy: ", np.mean(accuracy_list), "std: ", np.std(accuracy_list)
# print "f1: ", np.mean(f1_list), "std: ", np.std(f1_list)


# if __name__ == "__main__":
#     f = open('./data/preprocessing_data.p', 'rb')
#     loaded_data = []
#     for i in range(5): # [reverse_dictionary, train_sequence, test_sequence, train_label, test_label]:
#         loaded_data.append(cPickle.load(f))
#     f.close()
#
#     dictionary = loaded_data[0]
#     train_sequence = loaded_data[1]
#     test_sequence = loaded_data[2]
#     train_label = loaded_data[3]
#     test_label = loaded_data[4]
#
#     max_sequence_length = 20
#     train_sequence = pad_sequences(train_sequence, maxlen=max_sequence_length)
#     test_sequence = pad_sequences(test_sequence, maxlen=max_sequence_length)
#     f = open('./data/embedding_matrix.p')
#     embedding_matrix = cPickle.load(f)
#     f.close
#
#     embedding_size = embedding_matrix.shape[1]
#
#
#     # TODO -- train on matrix ?






