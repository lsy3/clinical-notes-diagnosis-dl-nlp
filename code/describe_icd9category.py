
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
df_diag_m = df_diag_m.withColumn("icd9_code", geticd9cat_udf("icd9_code"))
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

print df_ne.dtypes
print df_diag_m.dtypes
print df_diag_o.dtypes
print df_hadm_id_list.dtypes
print df_subject_id_list.dtypes

spark.sql("""
SELECT * FROM diagnoses_icd_m
LIMIT 10
""").show()

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
SELECT COUNT(DISTINCT subject_id), 
COUNT(DISTINCT hadm_id), COUNT(DISTINCT icd9_code)
FROM (
    SELECT row_id, subject_id, diagnoses_icd_m.hadm_id AS hadm_id,
    seq_num, icd9_code
    FROM diagnoses_icd_m JOIN (SELECT DISTINCT hadm_id FROM noteevents) AS a
    ON diagnoses_icd_m.hadm_id = a.hadm_id
)
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

print "Done!"
#sc.stop()
