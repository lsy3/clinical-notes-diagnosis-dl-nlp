
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
                       header=True,
                       schema=df_ne_struct)
df_ne.registerTempTable("noteevents")

# many icd to one hadm_id
diag_struct = StructType([StructField("ROW_ID", IntegerType(), True),
                          StructField("SUBJECT_ID", IntegerType(), True),
                          StructField("HADM_ID", IntegerType(), True),
                          StructField("SEQ_NUM", IntegerType(), True),
                          StructField("ICD9_CODE", StringType(), True)])
df_diag_m = spark.read.csv("./data/DIAGNOSES_ICD.csv",
                           header=True,
                           schema=df_diag_struct)
df_diag_m.registerTempTable("diagnoses_icd_m")

# one icd to one hadm_id (take the smallest seq number as primary)
diag_o_rdd = df_diag_m.rdd.sortBy(lambda x: (x.HADM_ID, x.SUBJECT_ID, x.SEQ_NUM)) \
    .groupBy(lambda x: x.HADM_ID) \
    .mapValues(list) \
    .reduceByKey(lambda x, y: x if x.SEQ_NUM < y.SEQ_NUM else y) \
    .map(lambda (hid, d): d[0])
df_diag_o = spark.createDataFrame(diag_o_rdd,
                                 schema=diag_struct)
df_diag_o.registerTempTable("diagnoses_icd_o")

# noteevents + many2one diagnoses_icd
df_ne_m = spark.sql("""
SELECT noteevents.subject_id AS subject_id, noteevents.hadm_id AS hadm_id,
noteevents.category AS category, noteevents.description AS description,
noteevents.iserror AS iserror, noteevents.text AS text,
diagnoses_icd_m.SEQ_NUM AS seq_num, diagnoses_icd_m.ICD9_CODE AS icd9_code
FROM noteevents
JOIN diagnoses_icd_m
ON noteevents.hadm_id = diagnoses_icd_m.hadm_id
AND noteevents.subject_id = diagnoses_icd_m.subject_id
""")
df_ne_m.registerTempTable("noteevents_m")

# noteevents + one2one diagnoses_icd
df_ne_o = spark.sql("""
SELECT noteevents.subject_id AS subject_id, noteevents.hadm_id AS hadm_id,
noteevents.category AS category, noteevents.description AS description,
noteevents.iserror AS iserror, noteevents.text AS text,
diagnoses_icd_o.SEQ_NUM AS seq_num, diagnoses_icd_o.ICD9_CODE AS icd9_code
FROM noteevents
JOIN diagnoses_icd_o
ON noteevents.hadm_id = diagnoses_icd_o.hadm_id
AND noteevents.subject_id = diagnoses_icd_o.subject_id
""")
df_ne_o.registerTempTable("noteevents_o")

print df_ne.dtypes
print df_diag_m.dtypes
print df_diag_o.dtypes
print df_ne_m.dtypes
print df_ne_o.dtypes

spark.sql("""
SELECT COUNT(*), COUNT(DISTINCT subject_id), COUNT(DISTINCT hadm_id)
FROM noteevents
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
COUNT(DISTINCT hadm_id), COUNT(DISTINCT ICD9_CODE)
FROM diagnoses_icd_o
""").show()

# check code
spark.sql("""
SELECT *
FROM diagnoses_icd_o
WHERE seq_num <> 1
""").show()

spark.sql("""
SELECT COUNT(*), COUNT(DISTINCT subject_id), 
COUNT(DISTINCT hadm_id), COUNT(DISTINCT icd9_code)
FROM noteevents_o
""").show()

spark.sql("""
SELECT icd9_code, COUNT(DISTINCT subject_id) AS sid_count
FROM noteevents_o
GROUP BY icd9_code
ORDER BY sid_count DESC
LIMIT 50
""").show(n=50)

spark.sql("""
SELECT icd9_code, COUNT(DISTINCT hadm_id) AS hadm_count
FROM noteevents_o
GROUP BY icd9_code
ORDER BY hadm_count DESC
LIMIT 50
""").show(n=50)

spark.sql("""
SELECT COUNT(*), COUNT(DISTINCT subject_id), 
COUNT(DISTINCT hadm_id), COUNT(DISTINCT icd9_code)
FROM noteevents_m
""").show()

spark.sql("""
SELECT icd9_code, COUNT(DISTINCT subject_id) AS sid_count
FROM noteevents_m
GROUP BY icd9_code
ORDER BY sid_count DESC
LIMIT 50
""").show(n=50)

spark.sql("""
SELECT icd9_code, COUNT(DISTINCT hadm_id) AS hadm_count
FROM noteevents_m
GROUP BY icd9_code
ORDER BY hadm_count DESC
LIMIT 50
""").show(n=50)

df_icd9score = spark.sql("""
SELECT icd9_code, COUNT(DISTINCT hadm_id) AS score
FROM noteevents_m
GROUP BY icd9_code
ORDER BY score DESC
""")
df_icd9score.registerTempTable("icd9_score")

spark.sql("""
SELECT * FROM icd9_score
LIMIT 50
""").show(n=50)

df_nedi_top10 = spark.sql("""
SELECT * FROM noteevents_m
WHERE icd9_code IN 
    (SELECT icd9_code FROM icd9_score LIMIT 10)
""")
df_nedi_top10.write.csv("./data/NOTEEVENTS-TOP10.csv",
                       header=True)
df_nedi_top50 = spark.sql("""
SELECT * FROM noteevents_m
WHERE icd9_code IN 
    (SELECT icd9_code FROM icd9_score LIMIT 50)
""")
df_nedi_top50.write.csv("./data/NOTEEVENTS-TOP50.csv",
                       header=True)

sc.stop()
