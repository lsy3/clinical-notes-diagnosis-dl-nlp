
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *

conf = SparkConf().setAppName("preprocess").setMaster("local")
sc = SparkContext.getOrCreate(conf)
spark = SparkSession.builder.master("local").appName("preprocess").getOrCreate()

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

testdf = read_csv("./data/DATA_TFIDF_HADM_TOP10.csv")
print testdf.count()
testdf.show()


