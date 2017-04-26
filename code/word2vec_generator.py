
#### Dependecies

#scipy = 0.17.0
#gensim = 1.0.1
#pandas = 0.19.2
#numpy = 1.11.2

#### I used DATA_HADM.csv created using feature_extraction_nonseq.ipynb


import pandas as pd
import pickle
import random


df_hadm_top10 = pd.read_csv("./data/DATA_HADM.csv", escapechar='\\')
ICD9CODES = pickle.load(open("./data/ICD9CODES.p", "r"))


def separate(seed, N):    
    idx=list(range(N))
    random.seed(seed)
    random.shuffle(idx)
    idx_train= idx[0:int(N*0.50)]
    idx_val= idx[int(N*0.50):int(N*0.75)]
    idx_test= idx[int(N*0.75):N]

    return idx_train, idx_val, idx_test


idx_train, idx_val, idx_test = separate(1234, df_hadm_top10.shape[0])
idx_join_train=idx_train + idx_val
#len(idx_join_train)

#create dataframe with only train set
df_hadm_top10_w2v=df_hadm_top10.iloc[idx_join_train].copy()

# Cleanning the data
# Light preprocesing done on purpose (so word2vec understand sentence structure)
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text.split()
    return text

token_review = list(df_hadm_top10_w2v['text'].apply(preprocessor))


# Apply word2vec

from gensim.models import Word2Vec
from gensim import utils
from time import time

# assumptions: window is 5 words left and right, eliminate words than dont occur in
# more than 10 docs, use 4 workers for a quadcore machine. Size is the size of vector
# negative=5 implies negative sampling and makes doc2vec faster to train
# sg=0 means CBOW architecture used. sg=1 means skip-gram is used
# model = Word2Vec(sentence, size=100, window=5, workers=4, min_count=5)

size = 300  #change to 100 and 600 to generate vectors with those dimensions

#instantiate our  model
model_w2v = Word2Vec(min_count=10, window=5, size=size, sample=1e-3, negative=5, workers=4, sg=0)

#build vocab over all reviews
model_w2v.build_vocab(token_review)

#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
Idx=list(range(len(token_review)))

t0 = time()
for epoch in range(5):
     random.shuffle(Idx)
     perm_sentences = [token_review[i] for i in Idx]
     model_w2v.train(perm_sentences)
     print(epoch)
    
elapsed=time() - t0
print("Time taken for Word2vec training: ", elapsed, "seconds.")


# saves the word2vec model to be used later.
#model_w2v.save('./model_word2vec_skipgram_300dim')

# open a saved word2vec model 
#import gensim
#model_w2v=gensim.models.Word2Vec.load('./model_word2vec')

# save the model in txt format
model_w2v.wv.save_word2vec_format('./data/model_word2vec_v2_300dim.txt', binary=False)

