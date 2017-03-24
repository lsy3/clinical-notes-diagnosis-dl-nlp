from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import cPickle


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


def convert_sequence_to_matrix(sequence, embedding_matrix):
    pass

if __name__ == "__main__":
    f = open('./data/preprocessing_data.p', 'rb')
    loaded_data = []
    for i in range(5): # [reverse_dictionary, train_sequence, test_sequence, train_label, test_label]:
        loaded_data.append(cPickle.load(f))
    f.close()

    dictionary = loaded_data[0]
    train_sequence = loaded_data[1]
    test_sequence = loaded_data[2]
    train_label = loaded_data[3]
    test_label = loaded_data[4]

    max_sequence_length = 20
    train_sequence = pad_sequences(train_sequence, maxlen=max_sequence_length)
    test_sequence = pad_sequences(test_sequence, maxlen=max_sequence_length)
    f = open('./data/embedding_matrix.p')
    embedding_matrix = cPickle.load(f)
    f.close

    embedding_size = embedding_matrix.shape[1]


    # TODO -- train on matrix ?






