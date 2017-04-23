
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
import pyspark.sql.functions as F

conf = SparkConf().setAppName("preprocess").setMaster("local")
sc = SparkContext.getOrCreate(conf)
spark = SparkSession.builder.master("local").appName("preprocess").getOrCreate()

ne_struct = StructType([StructField("row_id", IntegerType(), True),
                      StructField("subject_id", IntegerType(), True),
                      StructField("hadm_id", IntegerType(), True),
                      StructField("chartdate", DateType(), True),
                      StructField("category", StringType(), True),
                      StructField("description", StringType(), True),
                      StructField("cgid", IntegerType(), True),
                      StructField("iserror", IntegerType(), True),
                      StructField("text", StringType(), True)])
df_ne = spark.read.csv("./data/NOTEEVENTS-2.csv",
# df_ne = spark.read.csv("./data/NOTEEVENTS-2sample.csv",
                       header=True,
                       schema=ne_struct)
df_ne.registerTempTable("noteevents")
df_ne.filter(df_ne.category=="Discharge summary") \
    .registerTempTable("noteevents2")
    
# i want to cache noteevents, but it's too big

# many icd to one hadm_id
diag_struct = StructType([StructField("ROW_ID", IntegerType(), True),
                          StructField("SUBJECT_ID", IntegerType(), True),
                          StructField("HADM_ID", IntegerType(), True),
                          StructField("SEQ_NUM", IntegerType(), True),
                          StructField("ICD9_CODE", StringType(), True)])
df_diag_m = spark.read.csv("./data/DIAGNOSES_ICD.csv",
                           header=True,
                           schema=diag_struct) \
            .selectExpr("ROW_ID as row_id", 
                        "SUBJECT_ID as subject_id",
                        "HADM_ID as hadm_id",
                        "SEQ_NUM as seq_num",
                        "ICD9_CODE as icd9_code")
# added to filter out categories
geticd9cat_udf = F.udf(lambda x: str(x)[:3], StringType())
df_diag_m = df_diag_m.withColumn("icd9_cat", geticd9cat_udf("icd9_code"))
df_diag_m.registerTempTable("diagnoses_icd_m")
df_diag_m.cache()

# one icd to one hadm_id (take the smallest seq number as primary)
diag_o_rdd = df_diag_m.rdd.sortBy(lambda x: (x.hadm_id, x.subject_id, x.seq_num)) \
    .groupBy(lambda x: x.hadm_id) \
    .mapValues(list) \
    .reduceByKey(lambda x, y: x if x.seq_num < y.seq_num else y) \
    .map(lambda (hid, d): d[0])
df_diag_o = spark.createDataFrame(diag_o_rdd)
df_diag_o.registerTempTable("diagnoses_icd_o")
df_diag_o.cache()

# get hadm_id list in noteevents
df_hadm_id_list = spark.sql("""
SELECT DISTINCT hadm_id FROM noteevents2
""")
df_hadm_id_list.registerTempTable("hadm_id_list")
df_hadm_id_list.cache()

# get subject_id list in noteevents
df_subject_id_list = spark.sql("""
SELECT DISTINCT subject_id FROM noteevents2
""")
df_subject_id_list.registerTempTable("subject_id_list")
df_subject_id_list.cache()

df_icd9desc = spark.read.csv("./data/D_ICD_DIAGNOSES.csv",
                       header=True, inferSchema=True)
df_icd9desc.registerTempTable("diagnoses_icd_desc")

df_diag_o2 = spark.sql("""
SELECT row_id, subject_id, diagnoses_icd_o.hadm_id AS hadm_id,
seq_num, icd9_code, icd9_cat
FROM diagnoses_icd_o JOIN hadm_id_list
ON diagnoses_icd_o.hadm_id = hadm_id_list.hadm_id
""")
df_diag_o2.registerTempTable("diagnoses_icd_o2")
df_diag_o2.cache()

df_diag_m2 = spark.sql("""
SELECT row_id, subject_id, diagnoses_icd_m.hadm_id AS hadm_id,
seq_num, icd9_code, icd9_cat
FROM diagnoses_icd_m JOIN hadm_id_list
ON diagnoses_icd_m.hadm_id = hadm_id_list.hadm_id
""")
df_diag_m2.registerTempTable("diagnoses_icd_m2")
df_diag_m2.cache()

print df_ne.dtypes
print df_diag_m.dtypes
print df_diag_o.dtypes
print df_hadm_id_list.dtypes
print df_subject_id_list.dtypes
print df_icd9desc.dtypes
print df_diag_o2.dtypes
print df_diag_m2.dtypes

icd9code_score_hadm = spark.sql("""
SELECT icd9_code, COUNT(DISTINCT hadm_id) AS score
FROM diagnoses_icd_m2
GROUP BY icd9_code
""").rdd.cache()

icd9code_score_subj = spark.sql("""
SELECT icd9_code, COUNT(DISTINCT subject_id) AS score
FROM diagnoses_icd_m2
GROUP BY icd9_code
""").rdd.cache()

icd9cat_score_hadm = spark.sql("""
SELECT icd9_cat AS icd9_code, COUNT(DISTINCT hadm_id) AS score
FROM diagnoses_icd_m2
GROUP BY icd9_cat
""").rdd.cache()

icd9cat_score_subj = spark.sql("""
SELECT icd9_cat AS icd9_code, COUNT(DISTINCT subject_id) AS score
FROM diagnoses_icd_m2
GROUP BY icd9_cat
""").rdd.cache()

def get_id_to_topicd9(id_type, icdcode, topX):
    if id_type == "hadm_id" and icdcode:
        icd9_score = icd9code_score_hadm
    elif id_type == "hadm_id" and not icdcode:
        icd9_score = icd9cat_score_hadm
    elif id_type == "subject_id" and icdcode:
        icd9_score = icd9code_score_subj
    elif id_type == "subject_id" and not icdcode:
        icd9_score = icd9cat_score_subj
    else: #default
        icd9_score = icd9code_score_hadm
    
        
    icd9_topX2 = [i.icd9_code for i in icd9_score.takeOrdered(topX, key=lambda x: -x.score)]
    if not icdcode:
        icd9_topX2 = ['c'+str(i) for i in icd9_topX2]
    else:
        icd9_topX2 = [str(i) for i in icd9_topX2]
    icd9_topX = set(icd9_topX2)
    
    id_to_topicd9 = df_diag_m2.rdd \
        .map(lambda x: (x.hadm_id if id_type=="hadm_id" else x.subject_id, x.icd9_code if icdcode else 'c'+str(x.icd9_cat))) \
        .groupByKey() \
        .mapValues(lambda x: set(x) & icd9_topX) \
        .filter(lambda (x, y): y)
        
    return id_to_topicd9, list(icd9_topX2)

print get_id_to_topicd9("hadm_id", True, 50)
print get_id_to_topicd9("subject_id", True, 50)
print get_id_to_topicd9("hadm_id", False, 50)
print get_id_to_topicd9("subject_id", False, 50)

import re

def sparse2vec(mapper, data):
    out = [0] * len(mapper)
    if data != None:
        for i in data:
            out[mapper[i]] = 1
    return out
    
def get_id_to_texticd9(id_type, topX, stopwords=[]):
    def remstopwords(text):
        text = re.sub('\[\*\*[^\]]*\*\*\]', '', text)
        text = re.sub('<[^>]*>', '', text)
        text = re.sub('[\W]+', ' ', text.lower()) 
        text = re.sub(" \d+", " ", text)
        return " ".join([i for i in text.split() if i not in stopwords])
    
    id_to_topicd9code, topicd9code = get_id_to_topicd9(id_type, True, topX)
    id_to_topicd9cat, topicd9cat = get_id_to_topicd9(id_type, False, topX)
    topX2 = 2 * topX
    topicd9 = topicd9code+topicd9cat
    mapper = dict(zip(topicd9, range(topX2)))
    
    id_to_topicd9 = id_to_topicd9code.fullOuterJoin(id_to_topicd9cat) \
        .map(lambda (id_, (icd9code, icd9cat)): (id_, \
                                                 (icd9code if icd9code else set()) | \
                                                 (icd9cat if icd9cat else set())))
        
    ne_topX = df_ne.rdd \
        .filter(lambda x: x.category == "Discharge summary") \
        .map(lambda x: (x.hadm_id if id_type=="hadm_id" else x.subject_id, x.text)) \
        .groupByKey() \
        .mapValues(lambda x: " ".join(x)) \
        #.join(id_to_topicd9) \ # involve only data related to top10
        # involve all data, even those not related to top10
        .leftOuterJoin(id_to_topicd9) \
        .map(lambda (id_, (text, icd9)): \
             [id_]+sparse2vec(mapper, icd9)+[text if len(stopwords) == 0 else remstopwords(text)])
#              list(Vectors.sparse(topX, dict.fromkeys(map(lambda x: mapper[x], icd9), 1))))
            
    return spark.createDataFrame(ne_topX, ["id"]+topicd9+["text"]), topicd9

# get_id_to_texticd9("hadm_id", 10)[0].show()

import pickle

ICD9CODES = spark.sql("""
SELECT DISTINCT icd9_code FROM diagnoses_icd_m2
""").rdd.map(lambda x: x.icd9_code).collect()
ICD9CODES = [str(i).lower() for i in ICD9CODES]

pickle.dump(ICD9CODES, open( "./data/ICD9CODES.p", "wb" ))

import time
t0 = time.time()

df_id2texticd9, topicd9 = get_id_to_texticd9("hadm_id", 50)
df_id2texticd9.write.csv("./data/DATA_HADM", header=True)

print topicd9
print df_id2texticd9.count()
print time.time() - t0

df_id2texticd9.show()

import pickle

print topicd9[:10]
pickle.dump(topicd9[:10], open( "./data/ICD9CODES_TOP10.p", "wb" ))
print topicd9[:50]
pickle.dump(topicd9[:50], open( "./data/ICD9CODES_TOP50.p", "wb" ))
print topicd9[50:60]
pickle.dump(topicd9[50:60], open( "./data/ICD9CAT_TOP10.p", "wb" ))
print topicd9[50:]
pickle.dump(topicd9[50:], open( "./data/ICD9CAT_TOP50.p", "wb" ))

import time
from nltk.corpus import stopwords

t0 = time.time()
STOPWORDS_WORD2VEC = stopwords.words('english') + ICD9CODES

df_id2texticd9, topicd9 = get_id_to_texticd9("hadm_id", 50, stopwords=STOPWORDS_WORD2VEC)
df_id2texticd9.write.csv("./data/DATA_HADM_CLEANED", header=True)
df_id2texticd9.cache()

print topicd9
print df_id2texticd9.count()
print time.time() - t0
df_id2texticd9.show()

import pandas as pd
df = pd.read_csv("./data/DATA_HADM.csv", escapechar='\\')
print df.head()

spark.sql("""
SELECT icd9_code
FROM diagnoses_icd_m2
GROUP BY icd9_code
ORDER BY COUNT(DISTINCT hadm_id) DESC
LIMIT 10
""").show()
    
# id_to_topicd9, topicd9 = get_id_to_topicd9("hadm_id", 10)
# print id_to_topicd9.count()

# spark.sql("""
# SELECT COUNT(DISTINCT hadm_id) AS hadm_count
# FROM diagnoses_icd_m2
# WHERE icd9_code IN
#     (SELECT icd9_code
#     FROM diagnoses_icd_m2
#     GROUP BY icd9_code
#     ORDER BY COUNT(DISTINCT hadm_id) DESC
#     LIMIT 10)
# """).show()

#sc.stop()
print "Done!"
