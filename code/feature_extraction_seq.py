
# coding: utf-8

# # Feature Extraction for Deep Learning (WORD2VEC)

# * Input:
#   * ./data/ICD9CODES.p (pickle file of all ICD9 codes)
#   * ./data/ICD9CODES_TOP10.p (pickle file of top 10 ICD9 codes)
#   * ./data/ICD9CODES_TOP50.p (pickle file of top 50 ICD9 codes)
#   * ./data/ICD9CAT_TOP10.p (pickle file of top 10 ICD9 categories)
#   * ./data/ICD9CAT_TOP50.p (pickle file of top 50 ICD9 categories)
#   * ./data/TRAIN-VAL-TEST-HADMID.p (pickle file of train-val-test sets. each set contains a list of hadm_id)
#   * ./data/DATA_HADM.csv (contains top 50 icd9code, top 50 icd9cat, and clinical text for each admission, source for seqv0)
#   * ./data/DATA_HADM_CLEANED.csv (contains top 50 icd9code, top 50 icd9cat, and cleaned clinical text w/out stopwords for each admission, source for seqv1)
#   * ./data/model_word2vec_v2_*dim.txt (our custom word2vec model)
#   * ./data/bio_nlp_vec/PubMed-shuffle-win-*.txt (pre-trained bionlp word2vec model. convert from .bin to .txt using gensim)
# * Output:
#   * ./data/DATA_WORDSEQV[0/1]_HADM_TOP[10/10CAT/50/50CAT].p (pickle file of train-val-test data and label)
#   * ./data/DATA_WORDSEQV[0/1]_WORDINDEX.p (pickle of file of word sequence index)
# * Description: 
#   * All sequential feature extraction tried in the paper.
#   * WORDSEQV0 = seqv0 in the paper
#   * WORDSEQV1 = seqv1 in the paper
#   * word2vec_*dim = custom word2vec with * features in the paper
#   * PubMed-shuffle-win-*.txt = pre trained word2vec in the paper (bio*)

# ## Initialization

# In[1]:

import pandas as pd
import numpy as np
import pickle, cPickle

ICD9CODES = pickle.load(open("./data/ICD9CODES.p", "r"))
ICD9CODES_TOP10 = pickle.load(open("./data/ICD9CODES_TOP10.p", "r"))
ICD9CODES_TOP50 = pickle.load(open("./data/ICD9CODES_TOP50.p", "r"))
ICD9CAT_TOP10 = pickle.load(open("./data/ICD9CAT_TOP10.p", "r"))
ICD9CAT_TOP50 = pickle.load(open("./data/ICD9CAT_TOP50.p", "r"))


# In[4]:

from nltk.corpus import stopwords
print len(stopwords.words('english'))


# ## WORD2VEC_DL_V0

# In[6]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

STOPWORDS_WORD2VEC = stopwords.words('english') + ICD9CODES

def preprocessor_word2vec(text):
    text = re.sub('\[\*\*[^\]]*\*\*\]', '', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower()) 
    text = re.sub(" \d+", " ", text)
    
    return text

def create_WORD2VEC_DL_V0(df, max_sequence_len=600, inputCol='text'):
    texts = df[inputCol].apply(preprocessor_word2vec)
    #texts = df['text']  # list of text samples

    toke = Tokenizer()
    toke.fit_on_texts(texts)
    sequence = toke.texts_to_sequences(texts)

    ave_seq = [len(i) for i in sequence]
    print 1.0* sum(ave_seq) / len(ave_seq)
    
    word_index = toke.word_index
    reverse_word_index = dict(zip(word_index.values(), word_index.keys())) # dict e.g. {1:'the', 2:'a' ...}
    #index_list = word_index.values()

    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequence, maxlen=max_sequence_len)
    
    return data, word_index, reverse_word_index

def create_EmbeddingMatrix_V0(word_index, word2vec_model_path, remove_stopwords=True):

    embeddings_index = {}
    f = open(word2vec_model_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    
    if remove_stopwords:
        # Delete stopwords and ICD9 codes from pre-trained dictionary , 
        # so they will be zeros when we create embedding_matrix
        keys_updated = [word for word in embeddings_index.keys() if word not in STOPWORDS_WORD2VEC]
        index2word_set=set(keys_updated)
    else:
        index2word_set=set(embeddings_index.keys())
    
    EMBEDDING_DIM = embeddings_index.values()[0].size  # dimensions of the word2vec model

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in index2word_set: 
            #embedding_vector = embeddings_index.get(word)
        #if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embeddings_index.get(word)
            
    return embedding_matrix


# ## Actual Feature Extraction

# In[3]:

import random, cPickle

def separate(seed, N):    
    idx=list(range(N))
    random.seed(seed)
    random.shuffle(idx)
    idx_train= idx[0:int(N*0.50)]
    idx_val= idx[int(N*0.50):int(N*0.75)]
    idx_test= idx[int(N*0.75):N]

    return idx_train, idx_val, idx_test

def separate_2(df, hadmid_pickle):
    f = open(hadmid_pickle, 'rb')
    hadmid_train = cPickle.load(f)
    hadmid_val = cPickle.load(f)
    hadmid_test = cPickle.load(f)
    f.close()
    
    df2 = df.copy()
    df2['_idx'] = df2.index
    df2.set_index('id', inplace=True)
    
    idx_train = df2.loc[hadmid_train]['_idx'].tolist()
    idx_val = df2.loc[hadmid_val]['_idx'].tolist()
    idx_test = df2.loc[hadmid_test]['_idx'].tolist()
    
    return idx_train, idx_val, idx_test

def batch_output_pickle(df, data, reversemap, fname, labels, hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p'):
    idx_tuple = separate_2(df, hadmid_pickle)
    
    f = open(fname, 'wb')
    cPickle.dump(reversemap, f, protocol=cPickle.HIGHEST_PROTOCOL)
    for i in idx_tuple:
        cPickle.dump(data[i], f, protocol=cPickle.HIGHEST_PROTOCOL)
    for i in idx_tuple:
        cPickle.dump(df.loc[i][labels].values, f, protocol=cPickle.HIGHEST_PROTOCOL)        
    f.close()
    
def output_pickle(obj, fname):
    f = open(fname, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


# In[4]:

df = pd.read_csv("./data/DATA_HADM.csv", escapechar='\\')
print df.head()

data, word_index, reverse_word_index = create_WORD2VEC_DL_V0(df.copy(), max_sequence_len=2000)
output_pickle(word_index, "./data/DATA_WORDSEQV0_WORDINDEX.p")
batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV0_HADM_TOP10.p", ICD9CODES_TOP10)
batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV0_HADM_TOP50.p", ICD9CODES_TOP50)
batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV0_HADM_TOP10CAT.p", ICD9CAT_TOP10)
batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV0_HADM_TOP50CAT.p", ICD9CAT_TOP50)


# In[5]:

em = create_EmbeddingMatrix_V0(word_index, "./data/model_word2vec.txt", remove_stopwords=True)
output_pickle(em, "./data/EMBMATRIXV0_WORD2VEC.p")
em = create_EmbeddingMatrix_V0(word_index, "./data/model_word2vec_v2_100dim.txt", remove_stopwords=True)
output_pickle(em, "./data/EMBMATRIXV0_WORD2VEC_v2_100dim.p")
em = create_EmbeddingMatrix_V0(word_index, "./data/model_word2vec_v2_300dim.txt", remove_stopwords=True)
output_pickle(em, "./data/EMBMATRIXV0_WORD2VEC_v2_300dim.p")
em = create_EmbeddingMatrix_V0(word_index, "./data/model_word2vec_v2_600dim.txt", remove_stopwords=True)
output_pickle(em, "./data/EMBMATRIXV0_WORD2VEC_v2_600dim.p")


# In[6]:

df = pd.read_csv("./data/DATA_HADM_CLEANED.csv", escapechar='\\')
print df.head()

data, word_index, reverse_word_index = create_WORD2VEC_DL_V0(df.copy(), max_sequence_len=1500)
output_pickle(word_index, "./data/DATA_WORDSEQV1_WORDINDEX.p")
batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV1_HADM_TOP10.p", ICD9CODES_TOP10)
batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV1_HADM_TOP50.p", ICD9CODES_TOP50)
batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV1_HADM_TOP10CAT.p", ICD9CAT_TOP10)
batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV1_HADM_TOP50CAT.p", ICD9CAT_TOP50)


# In[7]:

em = create_EmbeddingMatrix_V0(word_index, "./data/model_word2vec.txt", remove_stopwords=True)
output_pickle(em, "./data/EMBMATRIXV1_WORD2VEC.p")
em = create_EmbeddingMatrix_V0(word_index, "./data/model_word2vec_v2_100dim.txt", remove_stopwords=True)
output_pickle(em, "./data/EMBMATRIXV1_WORD2VEC_v2_100dim.p")
em = create_EmbeddingMatrix_V0(word_index, "./data/model_word2vec_v2_300dim.txt", remove_stopwords=True)
output_pickle(em, "./data/EMBMATRIXV1_WORD2VEC_v2_300dim.p")
em = create_EmbeddingMatrix_V0(word_index, "./data/model_word2vec_v2_600dim.txt", remove_stopwords=True)
output_pickle(em, "./data/EMBMATRIXV1_WORD2VEC_v2_600dim.p")


# In[4]:

word_index = cPickle.load(open("./data/DATA_WORDSEQV0_WORDINDEX.p", "rb"))
em = create_EmbeddingMatrix_V0(word_index, "./data/bio_nlp_vec/PubMed-shuffle-win-30.txt", remove_stopwords=False)
output_pickle(em, "./data/EMBMATRIXV0_BIONLPVEC_Pubmedshufflewin30F.p")
em = create_EmbeddingMatrix_V0(word_index, "./data/bio_nlp_vec/PubMed-shuffle-win-2.txt", remove_stopwords=False)
output_pickle(em, "./data/EMBMATRIXV0_BIONLPVEC_Pubmedshufflewin02F.p")

word_index = cPickle.load(open("./data/DATA_WORDSEQV1_WORDINDEX.p", "rb"))
em = create_EmbeddingMatrix_V0(word_index, "./data/bio_nlp_vec/PubMed-shuffle-win-30.txt", remove_stopwords=False)
output_pickle(em, "./data/EMBMATRIXV1_BIONLPVEC_Pubmedshufflewin30F.p")
em = create_EmbeddingMatrix_V0(word_index, "./data/bio_nlp_vec/PubMed-shuffle-win-2.txt", remove_stopwords=False)
output_pickle(em, "./data/EMBMATRIXV1_BIONLPVEC_Pubmedshufflewin02F.p")


# ## Extra

# Basic Statistics

# In[8]:

df = pd.read_csv("./data/DATA_HADM.csv", escapechar='\\')

texts = df['text'].apply(preprocessor_word2vec)
#texts = df['text']  # list of text samples

toke = Tokenizer()
toke.fit_on_texts(texts)
sequence = toke.texts_to_sequences(texts)

seq_len = [len(i) for i in sequence]
print "mean: ", np.mean(seq_len)
print "median: ", np.median(seq_len)
print "max: ", np.max(seq_len)
print "min: ", np.min(seq_len)
print "90th percentile: ", np.percentile(seq_len, 90)
print "95th percentile: ", np.percentile(seq_len, 95)


# In[8]:

df = pd.read_csv("./data/DATA_HADM_CLEANED.csv", escapechar='\\')

texts = df['text'].apply(preprocessor_word2vec)
#texts = df['text']  # list of text samples

toke = Tokenizer()
toke.fit_on_texts(texts)
sequence = toke.texts_to_sequences(texts)

seq_len = [len(i) for i in sequence]
print "mean: ", np.mean(seq_len)
print "median: ", np.median(seq_len)
print "max: ", np.max(seq_len)
print "min: ", np.min(seq_len)
print "90th percentile: ", np.percentile(seq_len, 90)
print "95th percentile: ", np.percentile(seq_len, 95)


# Create ./data/TRAIN-VAL-TEST-HADMID.p

# In[20]:

import cPickle

train = pd.read_csv("./data/DATA_TFIDFV1_HADM_TOP10_train.csv", escapechar='\\')
val = pd.read_csv("./data/DATA_TFIDFV1_HADM_TOP10_val.csv", escapechar='\\')
test = pd.read_csv("./data/DATA_TFIDFV1_HADM_TOP10_test.csv", escapechar='\\')

train2 = train['id'].tolist()
val2 = val['id'].tolist()
test2 = test['id'].tolist()

f = open('./data/TRAIN-VAL-TEST-HADMID.p', 'wb')
for obj in [train2, val2, test2]:
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()


# In[2]:

import cPickle

datafile = "./data/DATA_WORDSEQV1_HADM_TOP10CAT.p"
f = open(datafile, 'rb')
loaded_data = []
for i in range(7): # [reverse_dictionary, train_sequence, test_sequence, train_label, test_label]:
    loaded_data.append(cPickle.load(f))
f.close()

dictionary = loaded_data[0]
train_sequence = loaded_data[1]
val_sequence = loaded_data[2]
test_sequence = loaded_data[3]
train_label = loaded_data[4]
val_label = loaded_data[5]
test_label = loaded_data[6]

print train_sequence[:5,:]
print train_label[:5,:]


# In[ ]:



