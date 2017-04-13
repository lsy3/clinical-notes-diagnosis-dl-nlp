
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *

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

print df_ne.dtypes
print df_diag_m.dtypes
print df_diag_o.dtypes
print df_hadm_id_list.dtypes
print df_subject_id_list.dtypes

df_diag_o2 = spark.sql("""
SELECT row_id, subject_id, diagnoses_icd_o.hadm_id AS hadm_id,
seq_num, icd9_code
FROM diagnoses_icd_o JOIN hadm_id_list
ON diagnoses_icd_o.hadm_id = hadm_id_list.hadm_id
""")
df_diag_o2.registerTempTable("diagnoses_icd_o2")
df_diag_o2.cache()

df_diag_m2 = spark.sql("""
SELECT row_id, subject_id, diagnoses_icd_m.hadm_id AS hadm_id,
seq_num, icd9_code
FROM diagnoses_icd_m JOIN hadm_id_list
ON diagnoses_icd_m.hadm_id = hadm_id_list.hadm_id
""")
df_diag_m2.registerTempTable("diagnoses_icd_m2")
df_diag_m2.cache()

spark.sql("""
SELECT COUNT(*), COUNT(DISTINCT subject_id), COUNT(DISTINCT hadm_id)
FROM noteevents
""").show()
spark.sql("""
SELECT COUNT(*), COUNT(DISTINCT subject_id), COUNT(DISTINCT hadm_id)
FROM noteevents2
""").show()

spark.sql("""
SELECT COUNT(DISTINCT hadm_id) AS hadm_count
FROM diagnoses_icd_m2
WHERE icd9_code IN
    (SELECT icd9_code
    FROM diagnoses_icd_m2
    GROUP BY icd9_code
    ORDER BY COUNT(DISTINCT hadm_id) DESC
    LIMIT 10)
""").show()

spark.sql("""
SELECT COUNT(DISTINCT hadm_id) AS hadm_count
FROM diagnoses_icd_m2
WHERE icd9_code IN
    (SELECT icd9_code
    FROM diagnoses_icd_m2
    GROUP BY icd9_code
    ORDER BY COUNT(DISTINCT hadm_id) DESC
    LIMIT 50)
""").show()

spark.sql("""
SELECT COUNT(DISTINCT hadm_id) AS hadm_count
FROM diagnoses_icd_m2
WHERE icd9_code IN
    (SELECT icd9_code
    FROM diagnoses_icd_m2
    GROUP BY icd9_code
    ORDER BY COUNT(DISTINCT hadm_id) DESC
    LIMIT 100)
""").show()

spark.sql("""
SELECT DISTINCT(category)
FROM noteevents
""").show()

spark.sql("""
SELECT COUNT(*), COUNT(DISTINCT subject_id), 
COUNT(DISTINCT hadm_id), COUNT(DISTINCT ICD9_CODE)
FROM diagnoses_icd_m
""").show()

spark.sql("""
SELECT COUNT(*), COUNT(DISTINCT subject_id), 
COUNT(DISTINCT hadm_id), COUNT(DISTINCT LOWER(ICD9_CODE))
FROM diagnoses_icd_m
""").show()

spark.sql("""
SELECT COUNT(*), COUNT(DISTINCT subject_id), 
COUNT(DISTINCT hadm_id), COUNT(DISTINCT ICD9_CODE)
FROM diagnoses_icd_o
""").show()

spark.sql("""
SELECT COUNT(*), COUNT(DISTINCT subject_id), 
COUNT(DISTINCT hadm_id), COUNT(DISTINCT LOWER(ICD9_CODE))
FROM diagnoses_icd_o
""").show()

# check code
spark.sql("""
SELECT *
FROM diagnoses_icd_o
WHERE seq_num <> 1
""").show()

spark.sql("""
SELECT COUNT(DISTINCT subject_id), 
COUNT(DISTINCT hadm_id), COUNT(DISTINCT icd9_code)
FROM diagnoses_icd_o2
""").show()

spark.sql("""
SELECT icd9_code, COUNT(DISTINCT subject_id) AS sid_count
FROM diagnoses_icd_o2
GROUP BY icd9_code
ORDER BY sid_count DESC
LIMIT 50
""").show(n=50)

spark.sql("""
SELECT icd9_code, COUNT(DISTINCT hadm_id) AS hadm_count
FROM diagnoses_icd_o2
GROUP BY icd9_code
ORDER BY hadm_count DESC
LIMIT 50
""").show(n=50)

spark.sql("""
SELECT COUNT(DISTINCT subject_id), 
COUNT(DISTINCT hadm_id), COUNT(DISTINCT icd9_code)
FROM diagnoses_icd_m2
""").show()

spark.sql("""
SELECT icd9_code, COUNT(DISTINCT subject_id) AS sid_count
FROM diagnoses_icd_m2
GROUP BY icd9_code
ORDER BY sid_count DESC
LIMIT 50
""").show(n=50)

spark.sql("""
SELECT icd9_code, COUNT(DISTINCT hadm_id) AS hadm_count
FROM diagnoses_icd_m2
GROUP BY icd9_code
ORDER BY hadm_count DESC
LIMIT 50
""").show(n=50)

icd9_score_hadm = spark.sql("""
SELECT icd9_code, COUNT(DISTINCT hadm_id) AS score
FROM diagnoses_icd_m2
GROUP BY icd9_code
""").rdd.cache()

icd9_score_subj = spark.sql("""
SELECT icd9_code, COUNT(DISTINCT subject_id) AS score
FROM diagnoses_icd_m2
GROUP BY icd9_code
""").rdd.cache()

def get_id_to_topicd9(id_type, topX):
    if id_type == "hadm_id":
        icd9_score = icd9_score_hadm
    else:
        icd9_score = icd9_score_subj
        
    icd9_topX = set([i.icd9_code for i in icd9_score.takeOrdered(topX, key=lambda x: -x.score)])
    
    id_to_topicd9 = df_diag_m2.rdd \
        .map(lambda x: (x.hadm_id if id_type=="hadm_id" else x.subject_id, x.icd9_code)) \
        .groupByKey() \
        .mapValues(lambda x: set(x) & icd9_topX) \
        .filter(lambda (x, y): y)
        
    return id_to_topicd9, list(icd9_topX)

# for i in get_id_to_topicd9("hadm_id", 10)[0].take(3):
#     print i
# for i in get_id_to_topicd9("subject_id", 50)[0].take(3):
#     print i

def sparse2vec(mapper, data):
    out = [0] * len(mapper)
    for i in data:
        out[mapper[i]] = 1
    return out

def get_id_to_texticd9(id_type, topX):
    id_to_topicd9, topicd9 = get_id_to_topicd9(id_type, topX)
    mapper = dict(zip(topicd9, range(topX)))
    
    ne_topX = df_ne.rdd \
        .filter(lambda x: x.category == "Discharge summary") \
        .map(lambda x: (x.hadm_id if id_type=="hadm_id" else x.subject_id, x.text)) \
        .groupByKey() \
        .mapValues(lambda x: " ".join(x)) \
        .join(id_to_topicd9) \
        .map(lambda (id_, (text, icd9)): \
             [id_, text]+sparse2vec(mapper, icd9))
#              list(Vectors.sparse(topX, dict.fromkeys(map(lambda x: mapper[x], icd9), 1))))
        
    return spark.createDataFrame(ne_topX, ["id", "text"]+topicd9), mapper

# get_id_to_texticd9("hadm_id", 10)[0].show()

from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StopWordsRemover

def create_TFIDF(sentenceData, inputCol="text", outputCol="features", minDocFreq=3, numFeatures=20):
    tokenizer = RegexTokenizer(pattern="[.:\s]+", inputCol=inputCol, outputCol="z_words")
    wordsData = tokenizer.transform(sentenceData)
    
    remover = StopWordsRemover(inputCol="z_words", outputCol="z_filtered")
    wordsDataFiltered = remover.transform(wordsData)
    
    hashingTF = HashingTF(inputCol="z_filtered", outputCol="z_rawFeatures", numFeatures=numFeatures)
    featurizedData = hashingTF.transform(wordsDataFiltered)
    # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="z_rawFeatures", outputCol=outputCol, minDocFreq=minDocFreq)
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    
    return rescaledData.drop("z_words", "z_filtered", "z_rawFeatures", inputCol)

from pyspark.mllib.util import Vectors
from pyspark.mllib.linalg import VectorUDT
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import DataType, StringType

def output_csv(df, path):
    udf = UserDefinedFunction(lambda x: Vectors.stringify(x), StringType())
    new_df = df.withColumn('features', udf(df.features))
    
    new_df.write.csv(path, header=True)
    
def read_csv(path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    
    udf = UserDefinedFunction(lambda x: Vectors.parse(x), VectorUDT())
    new_df = df.withColumn('features', udf(df.features))
    
    return new_df

df_id2texticd9, topicd9_mapper = get_id_to_texticd9("hadm_id", 10)
df_id2featurelabel = create_TFIDF(df_id2texticd9, numFeatures=40000)

print topicd9_mapper
print df_id2featurelabel.dtypes
df_id2featurelabel.show()

output_csv(df_id2featurelabel, "./data/DATA_TFIDF_HADM_TOP10")

testdf = read_csv("./data/DATA_TFIDF_HADM_TOP10")
print testdf.count()
testdf.show()

spark.sql("""
SELECT icd9_code
FROM diagnoses_icd_m2
GROUP BY icd9_code
ORDER BY COUNT(DISTINCT hadm_id) DESC
LIMIT 10
""").show()
    
id_to_topicd9, topicd9 = get_id_to_topicd9("hadm_id", 10)
print id_to_topicd9.count()

spark.sql("""
SELECT COUNT(DISTINCT hadm_id) AS hadm_count
FROM diagnoses_icd_m2
WHERE icd9_code IN
    (SELECT icd9_code
    FROM diagnoses_icd_m2
    GROUP BY icd9_code
    ORDER BY COUNT(DISTINCT hadm_id) DESC
    LIMIT 10)
""").show()

sc.stop()
