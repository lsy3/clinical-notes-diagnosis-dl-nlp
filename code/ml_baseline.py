
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *

conf = SparkConf().setAppName("preprocess").setMaster("local")
sc = SparkContext.getOrCreate(conf)
spark = SparkSession.builder.master("local").appName("preprocess").getOrCreate()

from pyspark.mllib.util import Vectors, MLUtils
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
    # https://spark.apache.org/docs/latest/ml-migration-guides.html
    new_df = MLUtils.convertVectorColumnsToML(df.withColumn('features', udf(df.features)))
    
    return new_df

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import StringType, IntegerType
import pyspark.sql.functions as F
import numpy as np

concat_udf = F.udf(lambda cols: float(int("".join([str(int(x)) for x in cols]), 2)), DoubleType())

def evaluate(df, labelCols, gettopX=-1, getfirstX=-1):
    labelCols2 = [i+"_pred" for i in labelCols]
    df.cache()
    
    r_list = {i: np.zeros((len(labelCols))) for i in ['accuracy', 'precision', 'recall', 'fmeasure']}
    for i in xrange(len(labelCols)):
        predandlabels = df.select(labelCols2[i], labelCols[i]).rdd \
                        .map(lambda x: (float(x[labelCols2[i]]), float(x[labelCols[i]])))
        metrics = MulticlassMetrics(predandlabels)

        # print metrics.confusionMatrix()
        r_list['accuracy'][i] = metrics.accuracy
        r_list['precision'][i] = metrics.precision(1.0)
        r_list['recall'][i] = metrics.recall(1.0)
        r_list['fmeasure'][i] = metrics.fMeasure(label=1.0)

    results = {}
    for m, rs in r_list.iteritems():
        results[m] = np.mean(rs)
        
    for code, num in [('top', gettopX), ('first', getfirstX)]:
        if num <= 0: continue
        
        if code == 'top':
            idx = np.argsort(np.nan_to_num(r_list['fmeasure']))[-num:]
        elif code == 'first':
            idx = xrange(num)
        
        for m, rs in r_list.iteritems():
            results['{0}_{1}'.format(m, code)] = np.mean(rs[idx])
            
    return results

def evaluate_em(df, labelCols, metrics=["f1", "weightedPrecision", "weightedRecall", "accuracy"]):
    evaluator = MulticlassClassificationEvaluator()
    labelCols2 = [i+"_pred" for i in labelCols]
    df2 = df.withColumn("_label", concat_udf(F.array(labelCols)))
    df2 = df2.withColumn("_pred", concat_udf(F.array(labelCols2)))
    
    output = {}
    for m in metrics:
        result = evaluator.evaluate(df2, {evaluator.metricName: m,
                                         evaluator.predictionCol: "_pred",
                                         evaluator.labelCol: "_label"})
        output[m] = result
        
    return output

from pyspark.ml.classification import LogisticRegression

class CustomLogisticRegression:
    def __init__(self):
        pass
    
    def fit(self, df, maxIter=100, regParam=0.0, featuresCol="features", ignoreCols=["id"]):
        self.featuresCol = featuresCol
        self.labelCols = df.columns
        self.labelCols.remove("features")
        for c in ignoreCols:
            self.labelCols.remove(c)
        self.models = []
        
        for c in self.labelCols:
            lr = LogisticRegression(featuresCol=featuresCol,
                                    labelCol=c,
                                    predictionCol=c+"_pred",
                                    probabilityCol=c+"_prob",
                                    rawPredictionCol=c+"_rpred",
                                    maxIter=maxIter,
                                    regParam=regParam,
                                    family="binomial")
            model = lr.fit(df)
            self.models.append(model)
            
    def predict(self, df):
        df_out = df
        for c, m in zip(self.labelCols, self.models):
            df_out = m.transform(df_out)
            
        return df_out
        
        

from pyspark.ml.classification import RandomForestClassifier

class CustomRandomForestClassifier:
    def __init__(self):
        pass
    
    def fit(self, df, maxDepth=5, maxBins=32, numTrees=20, regParam=0.0, featuresCol="features", ignoreCols=["id"]):
        self.featuresCol = featuresCol
        self.labelCols = df.columns
        self.labelCols.remove("features")
        for c in ignoreCols:
            self.labelCols.remove(c)
        self.models = []
        
        for c in self.labelCols:
            lr = RandomForestClassifier(featuresCol=featuresCol,
                                        labelCol=c,
                                        predictionCol=c+"_pred",
                                        probabilityCol=c+"_prob",
                                        rawPredictionCol=c+"_rpred",
                                        maxDepth=maxDepth,
                                        maxBins=maxBins,
                                        impurity="gini",
                                        numTrees=numTrees,
                                        seed=None)
            model = lr.fit(df)
            self.models.append(model)
            
    def predict(self, df):
        df_out = df
        for c, m in zip(self.labelCols, self.models):
            df_out = m.transform(df_out)
            
        return df_out

def print_latex(inum, m1, m2, m3, m4):
    r1 = "{precision:.4f} & {recall:.4f} & {fmeasure:.4f} & {accuracy:.4f}".format(**m1)
    r2 = "{precision:.4f} & {recall:.4f} & {fmeasure:.4f} & {accuracy:.4f}".format(**m2)
    r3 = "{accuracy:.4f}".format(**m3)
    r4 = "{accuracy:.4f}".format(**m4)
    return "{0} & {1} & {2} & {3} & {4} \\\\ \hline".format(inum, r1, r3, r2, r4)

def print_latex2(inum, m1, m2):
    r1 = "{precision_top:.4f} & {recall_top:.4f} & {fmeasure_top:.4f} & {accuracy_top:.4f}".format(**m1)
    r2 = "{precision_top:.4f} & {recall_top:.4f} & {fmeasure_top:.4f} & {accuracy_top:.4f}".format(**m2)
    return "{0} & {1} & & {2} & \\\\ \hline".format(inum, r1, r2)

def print_latex3(inum, m1, m2):
    r1 = "{precision_first:.4f} & {recall_first:.4f} & {fmeasure_first:.4f} & {accuracy_first:.4f}".format(**m1)
    r2 = "{precision_first:.4f} & {recall_first:.4f} & {fmeasure_first:.4f} & {accuracy_first:.4f}".format(**m2)
    return "{0} & {1} & & {2} & \\\\ \hline".format(inum, r1, r2)

def run_experiment(input_name, iterations=[5, 10, 25, 50, 75, 100], gettopX=-1, getfirstX=-1):
    df_train = read_csv("{0}_train.csv".format(input_name))
    df_val = read_csv("{0}_val.csv".format(input_name))
    df_test = read_csv("{0}_test.csv".format(input_name))

    #df_train = df_train.union(df_val)
    
    df_train.cache()
    df_test.cache()
    
    print input_name
    print "Train, Test:", df_train.count(), df_test.count()
    print "iter & train prec & recall & f1 & accuracy & em & test prec & recall & f1 & accuracy & em"
    for maxIter in iterations:
        clr = CustomLogisticRegression()
        clr.fit(df_train, maxIter=maxIter)
        df_pred_train = clr.predict(df_train)
        df_pred_test = clr.predict(df_test)

        r1 = evaluate(df_pred_train, clr.labelCols, gettopX=gettopX, getfirstX=getfirstX)
        r2 = evaluate(df_pred_test, clr.labelCols, gettopX=gettopX, getfirstX=getfirstX)
        r3 = evaluate_em(df_pred_train, clr.labelCols, metrics=["accuracy"])
        r4 = evaluate_em(df_pred_test, clr.labelCols, metrics=["accuracy"])
        
        print print_latex(maxIter, r1, r2, r3, r4)
        if gettopX > 0:
            print print_latex2(str(maxIter)+" top", r1, r2)
        if getfirstX > 0:
            print print_latex3(str(maxIter)+" first", r1, r2)



def run_experiment2(input_name, depths=[5, 10, 20, 30], gettopX=-1, getfirstX=-1):
    df_train = read_csv("{0}_train.csv".format(input_name))
    df_val = read_csv("{0}_val.csv".format(input_name))
    df_test = read_csv("{0}_test.csv".format(input_name))

    #df_train = df_train.union(df_val)
    
    df_train.cache()
    df_test.cache()
    
    print input_name
    print "Train, Test:", df_train.count(), df_test.count()
    print "iter & train prec & recall & f1 & accuracy & em & test prec & recall & f1 & accuracy & em"        
    for maxDepth in depths:
        clr = CustomRandomForestClassifier()
        clr.fit(df_train, maxDepth=maxDepth)
        df_pred_train = clr.predict(df_train)
        df_pred_test = clr.predict(df_test)

        r1 = evaluate(df_pred_train, clr.labelCols, gettopX=gettopX, getfirstX=getfirstX)
        r2 = evaluate(df_pred_test, clr.labelCols, gettopX=gettopX, getfirstX=getfirstX)
        r3 = evaluate_em(df_pred_train, clr.labelCols, metrics=["accuracy"])
        r4 = evaluate_em(df_pred_test, clr.labelCols, metrics=["accuracy"])
        
        print print_latex(maxDepth, r1, r2, r3, r4)
        if gettopX > 0:
            print print_latex2(str(maxDepth)+" top", r1, r2)
        if getfirstX > 0:
            print print_latex3(str(maxDepth)+" first", r1, r2)



run_experiment("./data/DATA_TFIDFV0_HADM_TOP10")
run_experiment("./data/DATA_TFIDFV1_HADM_TOP10")
run_experiment("./data/DATA_WORD2VECV0_HADM_TOP10")
run_experiment("./data/DATA_WORD2VECV1_HADM_TOP10")
run_experiment("./data/DATA_WORD2VECV2_HADM_TOP10")

run_experiment2("./data/DATA_TFIDFV0_HADM_TOP10")
run_experiment2("./data/DATA_TFIDFV1_HADM_TOP10")
run_experiment2("./data/DATA_WORD2VECV0_HADM_TOP10")
run_experiment2("./data/DATA_WORD2VECV1_HADM_TOP10")
run_experiment2("./data/DATA_WORD2VECV2_HADM_TOP10")

run_experiment("./data/DATA_TFIDFV0_HADM_TOP50")
run_experiment("./data/DATA_TFIDFV1_HADM_TOP50")
run_experiment("./data/DATA_WORD2VECV0_HADM_TOP50")
run_experiment("./data/DATA_WORD2VECV1_HADM_TOP50")
run_experiment("./data/DATA_WORD2VECV2_HADM_TOP50")

run_experiment2("./data/DATA_TFIDFV0_HADM_TOP50", depths=[5, 10, 20])
run_experiment2("./data/DATA_TFIDFV1_HADM_TOP50", depths=[5, 10, 20])
run_experiment2("./data/DATA_WORD2VECV0_HADM_TOP50", depths=[5, 10, 20])

run_experiment2("./data/DATA_WORD2VECV0_HADM_TOP50", depths=[20])
run_experiment2("./data/DATA_WORD2VECV1_HADM_TOP50", depths=[5, 10, 20])
run_experiment2("./data/DATA_WORD2VECV2_HADM_TOP50", depths=[5])

run_experiment2("./data/DATA_WORD2VECV2_HADM_TOP50", depths=[10, 20])

run_experiment("./data/DATA_TFIDFV0_HADM_TOP10CAT")
run_experiment("./data/DATA_TFIDFV1_HADM_TOP10CAT")
run_experiment("./data/DATA_WORD2VECV0_HADM_TOP10CAT")
run_experiment("./data/DATA_WORD2VECV1_HADM_TOP10CAT")
run_experiment("./data/DATA_WORD2VECV2_HADM_TOP10CAT")

run_experiment2("./data/DATA_TFIDFV0_HADM_TOP10CAT")
run_experiment2("./data/DATA_TFIDFV1_HADM_TOP10CAT")
run_experiment2("./data/DATA_WORD2VECV0_HADM_TOP10CAT")
run_experiment2("./data/DATA_WORD2VECV1_HADM_TOP10CAT")
run_experiment2("./data/DATA_WORD2VECV2_HADM_TOP10CAT")

run_experiment("./data/DATA_TFIDFV0_HADM_TOP50CAT")
run_experiment("./data/DATA_TFIDFV1_HADM_TOP50CAT")
run_experiment("./data/DATA_WORD2VECV0_HADM_TOP50CAT")
run_experiment("./data/DATA_WORD2VECV1_HADM_TOP50CAT")
run_experiment("./data/DATA_WORD2VECV2_HADM_TOP50CAT")

run_experiment2("./data/DATA_TFIDFV0_HADM_TOP50CAT", depths=[5, 10, 20])
run_experiment2("./data/DATA_TFIDFV1_HADM_TOP50CAT", depths=[5, 10, 20])
run_experiment2("./data/DATA_WORD2VECV0_HADM_TOP50CAT", depths=[5, 10])

run_experiment2("./data/DATA_WORD2VECV0_HADM_TOP50CAT", depths=[20])
run_experiment2("./data/DATA_WORD2VECV1_HADM_TOP50CAT", depths=[5, 10])

run_experiment2("./data/DATA_WORD2VECV1_HADM_TOP50CAT", depths=[20])
run_experiment2("./data/DATA_WORD2VECV2_HADM_TOP50CAT", depths=[5, 10, 20])

run_experiment2("./data/DATA_WORD2VECV2_HADM_TOP50CAT", depths=[20])

run_experiment("./data/DATA_TFIDFV1_HADM_TOP50", iterations=[10], gettopX=10, getfirstX=10)
run_experiment("./data/DATA_TFIDFV1_HADM_TOP50CAT", iterations=[10], gettopX=10, getfirstX=10)

run_experiment2("./data/DATA_TFIDFV1_HADM_TOP50", depths=[20], gettopX=10, getfirstX=10)
run_experiment2("./data/DATA_TFIDFV1_HADM_TOP50CAT", depths=[20], gettopX=10, getfirstX=10)

run_experiment("./data/DATA_WORD2VECV3_HADM_TOP10")
run_experiment("./data/DATA_WORD2VECV4_HADM_TOP10")
run_experiment2("./data/DATA_WORD2VECV3_HADM_TOP10")
run_experiment2("./data/DATA_WORD2VECV4_HADM_TOP10")

run_experiment("./data/DATA_DOC2VECV0_HADM_TOP10")
run_experiment("./data/DATA_DOC2VECV1_HADM_TOP10")
run_experiment("./data/DATA_DOC2VECV2_HADM_TOP10")
run_experiment2("./data/DATA_DOC2VECV0_HADM_TOP10")
run_experiment2("./data/DATA_DOC2VECV1_HADM_TOP10")
run_experiment2("./data/DATA_DOC2VECV2_HADM_TOP10")

print "Done!"


