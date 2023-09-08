from math import ceil

import findspark
import pandas
import pyspark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import monotonically_increasing_id, udf
from pyspark.sql.types import IntegerType

from darima.dlsa import dlsa_mapreduce
from darima.evaluation import model_eval
from darima.forecast import darima_forec
from mycode.splitted.darimamodel import darima_model_udf
from mycode.splitted.settings import file_train_path, horizon, amount_of_dist, period, level, file_test_path, \
    dist_id_name, \
    Y_name, id_name, X_name
from mycode.splitted.types import schema_sdf, usecoef_ar

# Init Spark
findspark.init()
conf = pyspark.SparkConf().setAppName("DARIMA").setExecutorEnv('ARROW_PRE_0_15_IPC_FORMAT', '1')
spark = SparkSession.builder.getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

# Read CSV as Spark DataFrame, Change column order, Remove na, Add id
y_t: DataFrame = spark.read.csv(file_train_path, header=True, schema=schema_sdf).limit(horizon).select(
    [X_name] + [Y_name]).dropna().withColumn(id_name, monotonically_increasing_id() + 1)

# Divide data on partitions
T_time_series_length: int = y_t.count()
n_series_per_dist: int = int(T_time_series_length / amount_of_dist)


def calc_dist_id(id_value):
    dist_nr = ceil(id_value / n_series_per_dist)
    if dist_nr > amount_of_dist:
        return amount_of_dist
    else:
        return dist_nr


udf_calc_dist_id = udf(calc_dist_id, IntegerType())

# Add partition_id, Select only demand- and partition_id-columns
y_t: DataFrame = y_t.withColumn(dist_id_name, udf_calc_dist_id(id_name)).select([Y_name] + [dist_id_name])

# Map: Calculate model
model_mapped_sdf: DataFrame = y_t.groupby(dist_id_name).apply(darima_model_udf)
print("-----------------------MODEL RESULTS--------------------------")
print(model_mapped_sdf.show(10))

# Redduce
Sig_Theta: pandas.DataFrame = dlsa_mapreduce(model_mapped_sdf, T_time_series_length)
print("------------------------REDUCE RESULTS--------------------------")
print(Sig_Theta.head())

# Forecast
train_data: pandas.DataFrame = y_t.toPandas()[Y_name]
out_Theta: pandas.Series = Sig_Theta["Theta_tilde"]
out_Sigma: pandas.Series = Sig_Theta[usecoef_ar]
out_model_forec: pandas.DataFrame = darima_forec(Theta=out_Theta, Sigma=out_Sigma,
                                                 x=train_data, period=period,
                                                 h=horizon, level=level)
print("------------------------FORECAST RESULTS--------------------------")
print(out_model_forec.head())

# Evaluation
test_data: DataFrame = spark.read.csv(file_test_path, header=True, schema=schema_sdf).limit(horizon).toPandas()[
    Y_name]
out_model_eval: pandas.DataFrame = model_eval(x=train_data, xx=test_data, period=period,
                                              pred=out_model_forec["pred"], lower=out_model_forec["lower"],
                                              upper=out_model_forec["upper"], level=level)
score = out_model_eval.mean(axis=0)
print("------------------------EVALUATION RESULTS--------------------------")
print(out_model_eval.head())
print("------------------------EVALUATION SCORE RESULTS--------------------------")
print(score)

spark.stop()
