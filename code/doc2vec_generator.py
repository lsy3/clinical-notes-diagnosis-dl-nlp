#### Dependecies

#scipy = 0.17.0
#gensim = 1.0.1
#pandas = 0.19.2
#numpy = 1.11.2

#### I used DATA_HADM.csv from feature_extraction_nonseq.ipynb 

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

#create a dataframe with only train set
df_hadm_top10_d2v=df_hadm_top10.iloc[idx_join_train].copy()


# Cleanning the data
# Light preprocesing done on purpose (so doc2vec understand sentence structure)

import re
import gensim

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text.split()
    return text

token_review = list(df_hadm_top10_d2v['text'].apply(preprocessor))

LabeledSentence = gensim.models.doc2vec.LabeledSentence
    
def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

sentence=labelizeReviews(token_review, "note")


# Apply doc2vec

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from gensim import utils
from time import time

# assumptions: window is 5 words left and right, eliminate words than dont occur in
# more than 10 docs, use 4 workers for a quadcore machine. Size is the size of vector
# negative=5 implies negative sampling and makes doc2vec faster to train
#model = Doc2Vec(sentence, size=100, window=5, workers=4, min_count=5)


size = 600 #change to 100 and 300 to generate vector with those dimensions

#instantiate our model
model_dm = Doc2Vec(min_count=10, window=5, size=size, sample=1e-3, negative=5, workers=4)

#build vocab over all reviews
model_dm.build_vocab(sentence)

#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
Idx=list(range(len(sentence)))

t0 = time()
for epoch in range(5):
     random.shuffle(Idx)
     perm_sentences = [sentence[i] for i in Idx]
     model_dm.train(perm_sentences)
     print(epoch)
    
elapsed=time() - t0
print("Time taken for Doc2vec training: ", elapsed, "seconds.")


# saves the doc2vec model to be used later.
#model_dm.save('./data/model_doc2vec_v2_600dim')

# open a saved doc2vec model 
model_dm=gensim.models.Doc2Vec.load('./data/model_doc2vec_v2_600dim')


# Create doc2vec vector for test set

import re
def preprocessor_2(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    #text = text.split()
    return text

test_set_clean=df_hadm_top10.iloc[idx_test]['text'].apply(preprocessor_2)


t0 = time()
model_dm_600dim=test_set_clean.apply(model_dm.infer_vector)
elapsed=time() - t0
print("Time taken for Doc2vec pred over test set: ", elapsed, "seconds.")


import numpy as np

train_d2v=np.asarray(model_dm.docvecs)
df_train_d2v = pd.DataFrame(data=train_d2v, index=df_hadm_top10.iloc[idx_join_train]['id'])
test_d2v = [tuple(x) for x in model_dm_600dim]
df_test_d2v = pd.DataFrame(data=test_d2v, index=df_hadm_top10.iloc[idx_test]['id'])
df_d2v = pd.concat([df_train_d2v,df_test_d2v])

#save to drive            
df_d2v.to_csv("./data/model_doc2vec_v2_600dim_final.csv")


