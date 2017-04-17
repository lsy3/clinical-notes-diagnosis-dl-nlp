import tensorflow as tf
import matplotlib.pyplot as plt
import cPickle

def plot_with_labels(low_dim_embs, labels, filename='./data/tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    dictionary_path = './data/dictionary_v0.p'
    f = open(dictionary_path, 'rb')
    #
    # loaded_data = []
    # for i in range(7): # [reverse_dictionary, train_sequence, test_sequence, train_label, test_label]:
    #     loaded_data.append(cPickle.load(f))
    # f.close()

    dictionary = cPickle.load(f)
    f.close()

    embedding_matrix_path = './data/EMBMATRIXV0_WORD2VEC_v2_100dim.p'
    f = open(embedding_matrix_path)
    embedding_matrix = cPickle.load(f)
    f.close()


    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 100
    low_dim_embs = tsne.fit_transform(embedding_matrix[:plot_only, :])
    labels = [dictionary[i] for i in xrange(1, plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings")