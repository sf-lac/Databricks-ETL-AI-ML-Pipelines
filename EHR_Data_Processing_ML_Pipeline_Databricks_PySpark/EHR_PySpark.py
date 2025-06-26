# Databricks notebook source
# MAGIC %md #Patient Treatment Classification
# MAGIC ### Electronic Health Record Dataset
# MAGIC The dataset contains patients' laboratory test results used to determine the next patient's treatment whether **in care** or **out care**.
# MAGIC 
# MAGIC The dataset was obtained from [kaggle](https://kaggle.com) with original data from *Sadikin, Mujiono (2020), “EHR Dataset for Patient Treatment Classification”, Mendeley Data, V1, doi: 10.17632/7kv3rctx7m.1*

# COMMAND ----------

# MAGIC %md ###Ingest Data into Databricks Notebooks
# MAGIC 
# MAGIC Create and query a table or DataFrame uploaded to DBFS. [DBFS](https://docs.databricks.com/dbfs/index.html) is **Databricks File System** that allows storage of data for querying inside of Databricks. 
# MAGIC 
# MAGIC **PySpark** package is automatically loaded in Databricks and used to read .csv data into a **Spark DataFrame**.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/data_ori.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

sdf = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(sdf)

# COMMAND ----------

sdf.show(3)

# COMMAND ----------

display(sdf.take(3))

# COMMAND ----------

sdf.printSchema()

# COMMAND ----------

sdf.columns

# COMMAND ----------

# MAGIC %fs ls dbfs:/user/hive/warehouse

# COMMAND ----------

# MAGIC %fs ls dbfs:/user/hive/warehouse/ehr/

# COMMAND ----------

# MAGIC %fs rm -r dbfs:/user/hive/warehouse/ehr/

# COMMAND ----------

# Create table from DataFrame
table_name = "ehr"

sdf.write.format("parquet").saveAsTable(table_name)

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select * from ehr limit 10;

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select count(*) from ehr;

# COMMAND ----------

sdf.count()

# COMMAND ----------

sdf2 = spark.table('ehr')
display(sdf2.describe())

# COMMAND ----------

from pyspark.sql.functions import mean, min, max
sdf.select([mean('AGE'), min('AGE'), max('AGE')]).show()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select avg(AGE) as meanAGE, SOURCE from ehr group by SOURCE;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select count(SOURCE) as treatment, SEX from ehr group by SEX;

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select * from ehr;

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-whitegrid')

# create local Pandas DataFrame
pdf = sdf.toPandas()
pdf.head(3)

# Pearson's Correlation of features w.r.t each other
corr_matt = pdf.corr(method='pearson')
plt.figure(figsize=(8,8))
corr = sns.heatmap(corr_matt, annot=True, cmap='Blues', cbar=False)

# COMMAND ----------

sdf_ml = sdf.withColumn('TREATMENT', sdf['SOURCE'].astype('int'))
sdf_ml.printSchema()

# COMMAND ----------

sdf_ml = sdf.withColumns({'TREATMENT': sdf['SOURCE'].astype('int'), 'SEX_': sdf['SEX'].astype('int')}) 
sdf_ml.printSchema()

# COMMAND ----------

sdf_ml.show(3)

# COMMAND ----------

from pyspark.sql.functions import when, col
sdf_ml1 = sdf_ml.withColumn('TREATMENT', when(col('SOURCE') == 'in', 1).otherwise(0))
sdf_ml1.show()

# COMMAND ----------

sdf_ml2 = sdf_ml1.withColumn('SEX_', when(col('SEX') == 'M', 1).otherwise(0))
sdf_ml2.show()

# COMMAND ----------

sdf_ml2 = sdf_ml2.drop('SEX')
sdf_ml2 = sdf_ml2.drop('SOURCE')
sdf_ml2 = sdf_ml2.withColumnRenamed('SEX_', 'SEX')
sdf_ml2.printSchema()

# COMMAND ----------

sdf_ml2.show()

# COMMAND ----------

# create local Pandas DataFrame
pdf = sdf_ml2.toPandas()
pdf.head(3)

# Pearson's Correlation of features w.r.t target
corr_matt = pdf.corr(method='pearson')[['TREATMENT']].sort_values(by='TREATMENT',ascending=False)
plt.figure(figsize=(3,5))
corr = sns.heatmap(corr_matt, annot=True, cmap='Blues', cbar=False)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# vector assembled set of features
assembler = VectorAssembler(inputCols=['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE','THROMBOCYTE', 'AGE'], 
                            outputCol='features')
output = assembler.transform(sdf_ml2)
output.show(3)

# COMMAND ----------

output.select(['features']).take(1)

# COMMAND ----------

from pyspark.ml.feature import MinMaxScaler

scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
# rescale each feature to range [0,1]
scaled_data = scaler.fit(output).transform(output)
scaled_data.show(3)

# COMMAND ----------

scaled_data.select(['scaled_features']).take(1)

# COMMAND ----------

assembler = VectorAssembler(inputCols=['scaled_features', 'SEX'], 
                            outputCol='all_features')
output = assembler.transform(scaled_data)
output.show(3)

# COMMAND ----------

output.select(['all_features']).take(1)

# COMMAND ----------

sdf_ml_final = output.select('all_features', 'TREATMENT')
sdf_ml_final.printSchema()

# COMMAND ----------

train_data, test_data = sdf_ml_final.randomSplit([0.8, 0.2], seed=100)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

rfc = RandomForestClassifier(labelCol='TREATMENT',featuresCol='all_features')

# evaluation of model performance
evaluator = BinaryClassificationEvaluator(labelCol='TREATMENT', metricName='areaUnderROC')

# random forest parameters used in grid search-based model selection
param_grid = ParamGridBuilder().baseOn({rfc.seed : 100, rfc.maxBins : 64}
                                      ).addGrid(rfc.bootstrap, [True, False]
                                               ).addGrid(rfc.maxDepth, [2, 5, 10, 20]
                                                        ).addGrid(rfc.numTrees, [50, 100, 150]
                                                                 ).build()

# validation for hyper-parameter tuning of model performance during grid search
# TrainValidationSplit randomly splits the input dataset into train and validation sets, and uses evaluation metric on the validation set to select the best model
tvs = TrainValidationSplit(estimator = rfc,
                           estimatorParamMaps = param_grid,
                           evaluator = evaluator,
                           trainRatio = 0.8, 
                           seed =100)

# COMMAND ----------

rfc_model = tvs.fit(train_data)

# COMMAND ----------

rfc_model.bestModel

# COMMAND ----------

auc_predictions = rfc_model.transform(train_data)
auc_train_accuracy = evaluator.evaluate(auc_predictions, {evaluator.metricName:'areaUnderROC'})

# COMMAND ----------

print("AUC Train Accuracy = %g" % (auc_train_accuracy))

# COMMAND ----------

predictions = rfc_model.transform(train_data)
train_accuracy = evaluator.evaluate(predictions)
print("Training Accuracy = %g" % (train_accuracy))
print("Training Error = %g" % (1.0 - train_accuracy))

# COMMAND ----------

test_predictions = rfc_model.transform(test_data)
test_accuracy = evaluator.evaluate(test_predictions)
print("Test Accuracy = %g" % (test_accuracy))
print("Test Error = %g" % (1.0 - test_accuracy))

# COMMAND ----------

evaluator.metricName

# COMMAND ----------

rfc_model.evaluator

# COMMAND ----------

test_predictions.printSchema()

# COMMAND ----------

test_predictions.collect()
