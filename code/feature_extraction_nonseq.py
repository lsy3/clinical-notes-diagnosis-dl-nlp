
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
import pandas as pd
import pickle

ICD9CODES = pickle.load(open("./data/ICD9CODES.p", "r"))
ICD9CODES_TOP10 = pickle.load(open("./data/ICD9CODES_TOP10.p", "r"))
ICD9CODES_TOP50 = pickle.load(open("./data/ICD9CODES_TOP50.p", "r"))
ICD9CAT_TOP10 = pickle.load(open("./data/ICD9CAT_TOP10.p", "r"))
ICD9CAT_TOP50 = pickle.load(open("./data/ICD9CAT_TOP50.p", "r"))

from pyspark.ml.feature import StopWordsRemover
STOPWORDS_v0 = StopWordsRemover.loadDefaultStopWords("english") + ICD9CODES
STOPWORDS_v0 = [str(i) for i in STOPWORDS_v0]

# print "TFIDF v0 stop words"
# print STOPWORDS_v0

from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StopWordsRemover

def create_TFIDF_v0(trainData, applyData, inputCol="text", outputCol="features", minDocFreq=3, numFeatures=20):    
    tokenizer = RegexTokenizer(pattern="[.:\s]+", inputCol=inputCol, outputCol="z_words")
    wordsData1 = tokenizer.transform(trainData)
    wordsData2 = tokenizer.transform(applyData)
    
    remover = StopWordsRemover(inputCol="z_words", outputCol="z_filtered", stopWords=STOPWORDS_v0)
    wordsDataFiltered1 = remover.transform(wordsData1)
    wordsDataFiltered2 = remover.transform(wordsData2)
    
    hashingTF = HashingTF(inputCol="z_filtered", outputCol="z_rawFeatures", numFeatures=numFeatures)
    featurizedData1 = hashingTF.transform(wordsDataFiltered1)
    featurizedData2 = hashingTF.transform(wordsDataFiltered2)
    # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="z_rawFeatures", outputCol=outputCol, minDocFreq=minDocFreq)
    idfModel = idf.fit(featurizedData1)
    
    rescaledData = idfModel.transform(featurizedData2)
    return rescaledData.drop("z_words", "z_filtered", "z_rawFeatures", inputCol)

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

STOPWORDS_v1 = list(ENGLISH_STOP_WORDS) + ICD9CODES

# print "TFIDF v1 stop words"
# print STOPWORDS_v1

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.mllib.util import Vectors
from pyspark.mllib.linalg import VectorUDT
from pyspark.sql.functions import UserDefinedFunction

def preprocessor_v1(text):
    text = re.sub('\[\*\*[^\]]*\*\*\]', '', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower()) 
    return text

def create_TFIDF_v1(df_train, df_apply, inputCol="text", outputCol="features",
                    minDocFreq=3, maxDocFreq=1.0, numFeatures=20):
    df_train['z_cleaned'] = df_train[inputCol].apply(preprocessor_v1)
    df_apply['z_cleaned'] = df_apply[inputCol].apply(preprocessor_v1)

    # Now we create the sparse matrix of tfidf values
    tfidf = TfidfVectorizer(input='content',ngram_range=(1, 1),
                            stop_words=STOPWORDS_v1, 
                            min_df=minDocFreq,
                            max_df=maxDocFreq,
                            max_features=numFeatures)
    # I select to remove stopwords and minimun doc frequency =10 to delete very unusual words
    # that only show up in less than 10 notes (out of 59k notes available) 

    tfidf.fit([c for c in df_train['z_cleaned']])
    dtm = tfidf.transform([c for c in df_apply['z_cleaned']]).tocsr()
    dtm.sort_indices()
    df_apply[outputCol] = list(dtm)
   
    del df_train['z_cleaned']
    del df_apply['z_cleaned']
    del df_apply[inputCol]
    
    return df_apply

from nltk.corpus import stopwords

STOPWORDS_WORD2VEC = stopwords.words('english') + ICD9CODES

# print "WORD2VEC stop words"
# print STOPWORDS_WORD2VEC

import numpy as np
import re

# Run this cell if you are using Glove type format
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def preprocessor_word2vec(text):
    text = re.sub('\[\*\*[^\]]*\*\*\]', '', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower()) 
    text = re.sub(" \d+", " ", text)
    #text = gensim.parsing.preprocessing.remove_stopwords(text)
    return text

def makeFeatureVec(words, model, num_features, index2word_set):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    #index2word_set = set(model.wv.index2word) #activate if using gensim

    # activate if uploaded text version
    #index2word_set=set(keys_updated)
    
    
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, index2word_set, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 10000th review
       if counter%10000 == 0:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features,index2word_set)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs

def create_WORD2VEC(df, inputCol="text", outputCol="features",
                    word2vecmodel="./data/model_word2vec.txt"):
    df['z_cleaned'] = df[inputCol].apply(preprocessor_word2vec)
    
    # Create tokens
    token_review=[]
    for i in range(df['z_cleaned'].shape[0]):
        review = df['z_cleaned'][i]
        token_review.append([i for i in review.split()])
    
    model_w2v = loadGloveModel(word2vecmodel)
    numFeatures = len(model_w2v.values()[0])
    print "numFeatures: ", numFeatures
    
    keys_updated = [word for word in model_w2v.keys() if word not in STOPWORDS_WORD2VEC]
    index2word_set=set(keys_updated)

    final_w2v = getAvgFeatureVecs(token_review, model_w2v, index2word_set, num_features=numFeatures)
    df[outputCol] = list(final_w2v)
    
    del df['z_cleaned']
    del df[inputCol]
    
    return df

def create_DOC2VEC(df,doc2vecmodel):
    import pandas as pd
    import numpy as np
    df1=pd.read_csv(doc2vecmodel, index_col='id') 
    df1['features']=df1.values.tolist()
    df1=df1['features'].apply(np.asarray)
        
    result = pd.merge(pd.DataFrame({'id':df1.index, 'features':df1.values}), df, on='id')
    del result['text']
    del df1
    
    return result

import random, cPickle
import pandas as pd
from pyspark.mllib.util import Vectors
from pyspark.mllib.linalg import VectorUDT
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import StringType

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

def output_csv(df, path, col='features', dense=False):
    if type(df) != pd.DataFrame:       
        udf = UserDefinedFunction(lambda x: Vectors.stringify(x), StringType())
        df2 = df.withColumn(col, udf(df[col]))
        # df2.write.csv(path, header=True)
        
        df3 = df2.toPandas()
        df3.to_csv(path, index=False)
    else:
        N = df[col].iloc[0].shape[-1]
        if dense:
            def to_string(x):
                return "({0},[{1}],[{2}])".format(N, 
                                                  ",".join([str(i) for i in xrange(N)]),
                                                  ",".join([str(i) for i in x.tolist()]))
        else:            
            def to_string(x):
                return "({0},[{1}],[{2}])".format(N, 
                                      ",".join([str(i) for i in x.indices.tolist()]),
                                      ",".join([str(i) for i in x.data.tolist()]))
        df2 = df.copy()
        df2[col] = df[col].apply(to_string)
        df2.to_csv(path, index=False)

def batch_output_csv(df, otype, fname, labels, outputCol='features', 
                     hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p'):
    labels2 = ['id'] + labels + [outputCol]
        
    if otype.lower() == "tfidfv0":
        f = open(hadmid_pickle, 'rb')
        train_id_df = spark.createDataFrame(zip(cPickle.load(f)), ['id2'])
        val_id_df = spark.createDataFrame(zip(cPickle.load(f)), ['id2'])
        test_id_df = spark.createDataFrame(zip(cPickle.load(f)), ['id2'])
        f.close()
        
        df.cache()

        df1 = df.join(train_id_df, train_id_df.id2 == df.id, 'inner').select(labels2)
        output_csv(df1, "{0}_train.csv".format(fname))
        df1 = df.join(val_id_df, val_id_df.id2 == df.id, 'inner').select(labels2)
        output_csv(df1, "{0}_val.csv".format(fname))
        df1 = df.join(test_id_df, test_id_df.id2 == df.id, 'inner').select(labels2)
        output_csv(df1, "{0}_test.csv".format(fname))
        
    elif otype.lower() == "tfidfv1":
        idx_train, idx_val, idx_test = separate_2(df, hadmid_pickle)
        
        output_csv(df.loc[idx_train][labels2], "{0}_train.csv".format(fname), dense=False)
        output_csv(df.loc[idx_val][labels2], "{0}_val.csv".format(fname), dense=False)
        output_csv(df.loc[idx_test][labels2], "{0}_test.csv".format(fname), dense=False)
    elif otype.lower() == "word2vecv0":
        idx_train, idx_val, idx_test = separate_2(df, hadmid_pickle)
        output_csv(df.loc[idx_train][labels2], "{0}_train.csv".format(fname), dense=True)
        output_csv(df.loc[idx_val][labels2], "{0}_val.csv".format(fname), dense=True)
        output_csv(df.loc[idx_test][labels2], "{0}_test.csv".format(fname), dense=True)
    elif otype.lower() == "doc2vecv0":
        # doc2vec has the same format as word2vec
        idx_train, idx_val, idx_test = separate_2(df, hadmid_pickle)
        output_csv(df.loc[idx_train][labels2], "{0}_train.csv".format(fname), dense=True)
        output_csv(df.loc[idx_val][labels2], "{0}_val.csv".format(fname), dense=True)
        output_csv(df.loc[idx_test][labels2], "{0}_test.csv".format(fname), dense=True)
        
def read_csv(path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    
    udf = UserDefinedFunction(lambda x: Vectors.parse(x), VectorUDT())
    new_df = df.withColumn('features', udf(df.features))
    
    return new_df

df1_pd = pd.read_csv("./data/DATA_HADM.csv", escapechar='\\')
df1_sp = spark.read.csv("./data/DATA_HADM.csv", header=True, inferSchema=True)

idx_train, idx_val, idx_test = separate_2(df1_pd, './data/TRAIN-VAL-TEST-HADMID.p')
df1_pd_train = df1_pd.loc[idx_train]

f = open('./data/TRAIN-VAL-TEST-HADMID.p', 'rb')
hadmid_train = cPickle.load(f)
hadmid_val = cPickle.load(f)
hadmid_test = cPickle.load(f)
f.close()
    
hadmid_train_df = spark.createDataFrame(zip(hadmid_train), ['id2'])
df1_sp_train = df1_sp.join(hadmid_train_df, hadmid_train_df.id2 == df1_sp.id, 'inner')

print df1_pd.head()
print df1_pd_train.head()
print df1_sp.count()
print df1_sp_train.count()
df1_sp.show()
df1_sp_train.show()

from time import time
t0 = time()

df2 = create_TFIDF_v1(df1_pd_train.copy(), df1_pd.copy(), minDocFreq=10, 
                      maxDocFreq=0.8, numFeatures=40000)

batch_output_csv(df2, "tfidfv1", "./data/DATA_TFIDFV1_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "tfidfv1", "./data/DATA_TFIDFV1_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "tfidfv1", "./data/DATA_TFIDFV1_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "tfidfv1", "./data/DATA_TFIDFV1_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

from time import time
t0 = time()

df2 = create_WORD2VEC(df1_pd.copy(),
                      word2vecmodel="./data/model_word2vec_v2_100dim.txt")

batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV0_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV0_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV0_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV0_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

from time import time
t0 = time()

df2 = create_WORD2VEC(df1_pd.copy(),
                      word2vecmodel="./data/model_word2vec_v2_300dim.txt")

batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV1_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV1_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV1_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV1_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

from time import time
t0 = time()

df2 = create_WORD2VEC(df1_pd.copy(),
                      word2vecmodel="./data/model_word2vec_v2_600dim.txt")

batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV2_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV2_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV2_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV2_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

from time import time
t0 = time()

df2 = create_WORD2VEC(df1_pd.copy(),
                      word2vecmodel="./data/bio_nlp_vec/PubMed-shuffle-win-2.txt")

batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV3_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV3_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV3_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV3_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

from time import time
t0 = time()

df2 = create_WORD2VEC(df1_pd.copy(),
                      word2vecmodel="./data/bio_nlp_vec/PubMed-shuffle-win-30.txt")

batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV4_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV4_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV4_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "word2vecv0", "./data/DATA_WORD2VECV4_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

from time import time
t0 = time()

df2 = create_DOC2VEC(df1_pd.copy(),
                     doc2vecmodel="./data/model_doc2vec_v2_100dim_final.csv")

batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV0_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV0_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV0_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV0_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

from time import time
t0 = time()

df2 = create_DOC2VEC(df1_pd.copy(),
                     doc2vecmodel="./data/model_doc2vec_v2_300dim_final.csv")

batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV1_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV1_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV1_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV1_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

from time import time
t0 = time()

df2 = create_DOC2VEC(df1_pd.copy(),
                     doc2vecmodel="./data/model_doc2vec_v2_600dim_final.csv")

batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV2_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV2_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV2_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "doc2vecv0", "./data/DATA_DOC2VECV2_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

from time import time
t0 = time()

df2 = create_TFIDF_v0(df1_sp_train, df1_sp, numFeatures=40000)
print df2.count()

batch_output_csv(df2, "tfidfv0", "./data/DATA_TFIDFV0_HADM_TOP10", ICD9CODES_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "tfidfv0", "./data/DATA_TFIDFV0_HADM_TOP50", ICD9CODES_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "tfidfv0", "./data/DATA_TFIDFV0_HADM_TOP10CAT", ICD9CAT_TOP10,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')
batch_output_csv(df2, "tfidfv0", "./data/DATA_TFIDFV0_HADM_TOP50CAT", ICD9CAT_TOP50,
                 hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p')

elapsed=time() - t0
print("Run Time: ", elapsed, "seconds.")

tests = ["./data/DATA_TFIDFV0_HADM_TOP10",
        "./data/DATA_TFIDFV1_HADM_TOP10",
        "./data/DATA_WORD2VECV0_HADM_TOP10",
        "./data/DATA_WORD2VECV1_HADM_TOP10",
        "./data/DATA_WORD2VECV2_HADM_TOP10"]

for append in ["_train.csv", "_val.csv", "_test.csv"]:
    for folder in tests:
        fname = folder+append
        testdf = read_csv(fname)
        print fname
        print testdf.count()
        testdf.show()

#sc.stop()
print "Done!"


