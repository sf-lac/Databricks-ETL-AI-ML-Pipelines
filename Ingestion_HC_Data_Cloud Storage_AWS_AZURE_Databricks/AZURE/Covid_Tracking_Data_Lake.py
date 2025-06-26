# Databricks notebook source
# get COVID-19 Data Lake - Azure Open COVID Tracking Project Dataset 
# provides the numbers on tests, confirmed cases, hospitalizations, and patient outcomes from every US state and territory

# Azure storage access info
blob_account_name = "pandemicdatalake"
blob_container_name = "public"
blob_relative_path = "curated/covid-19/covid_tracking/latest/covid_tracking.parquet"
blob_sas_token = r""

# COMMAND ----------

# Allow SPARK to read from Blob remotely
wasbs_path = 'wasbs://%s@%s.blob.core.windows.net/%s' % (blob_container_name, blob_account_name, blob_relative_path)
spark.conf.set(
  'fs.azure.sas.%s.%s.blob.core.windows.net' % (blob_container_name, blob_account_name),
  blob_sas_token)
print('Remote blob path: ' + wasbs_path)

# COMMAND ----------

# SPARK read parquet
sdf = spark.read.parquet(wasbs_path)
print('Register the DataFrame as a SQL temporary view: dl_source')
sdf.createOrReplaceTempView('dl_source')

# COMMAND ----------

sdf.printSchema()

# COMMAND ----------

# Display top 10 rows
print('Displaying top 10 rows: ')
display(spark.sql('SELECT * FROM dl_source LIMIT 10'))

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DROP DATABASE IF EXISTS covid_tracking_dashboard CASCADE;
# MAGIC CREATE DATABASE IF NOT EXISTS covid_tracking_dashboard;
# MAGIC USE covid_tracking_dashboard;
# MAGIC

# COMMAND ----------

sdf.write.format("delta").mode("overwrite").save("/mnt/delta/covid_tracking")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS covid_tracking USING DELTA LOCATION '/mnt/delta/covid_tracking';

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM covid_tracking LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT count(*) FROM covid_tracking;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT min(date) as start, max(date) as end FROM covid_tracking;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM covid_tracking where state='WA';

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT DISTINCT state FROM covid_tracking;

# COMMAND ----------

display(spark.sql("desc detail covid_tracking_dashboard.covid_tracking"))

# COMMAND ----------

# MAGIC %fs ls dbfs:/mnt/delta/covid_tracking/

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT sum(hospitalized) as hospitalized, state FROM covid_tracking GROUP BY state;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT sum(hospitalized) as hospitalized, state, date_format(date, 'yyyy-MM') as period
# MAGIC FROM covid_tracking
# MAGIC WHERE hospitalized is not null 
# MAGIC GROUP BY period, state

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT sum(hospitalized) as hospitalized, sum(death) as deaths, sum(positive) as positives, date_format(date, 'yyyy-MM') as period
# MAGIC FROM covid_tracking 
# MAGIC WHERE state = 'WA'
# MAGIC GROUP BY period
# MAGIC ORDER BY period ASC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT sum(hospitalized) as hospitalized, sum(death) as deaths, sum(positive) as positives, state
# MAGIC FROM covid_tracking 
# MAGIC WHERE state in ('WA', 'FL', 'WI', 'AR')
# MAGIC GROUP BY state
