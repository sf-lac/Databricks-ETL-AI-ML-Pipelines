# Databricks notebook source
# MAGIC %md
# MAGIC Imports

# COMMAND ----------

import dlt
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC Download dataset and store in a Unity Catalog volume

# COMMAND ----------

catalog = 'main'
databaseName = 'default'
volumeName = 'unity_volume'
spark.sql("CREATE VOLUME " + catalog + "." + databaseName + "." + volumeName)

# COMMAND ----------

import os

os.environ["UNITY_CATALOG_VOLUME_PATH"] = "/Volumes/main/default/unity_volume/"
os.environ["DATASET_DOWNLOAD_URL"] = "https://health.data.ny.gov/api/views/jxy9-yhdk/rows.csv"
os.environ["DATASET_DOWNLOAD_FILENAME"] = "rows.csv"

dbutils.fs.cp(f"{os.environ.get('DATASET_DOWNLOAD_URL')}", f"{os.environ.get('UNITY_CATALOG_VOLUME_PATH')}{os.environ.get('DATASET_DOWNLOAD_FILENAME')}")

# COMMAND ----------

# MAGIC %md
# MAGIC Ingest raw data into a bronze (raw) table

# COMMAND ----------

@dlt.table(
  comment="Data ingested from NY State Department of Health"
)
def dlt_bronze():
  df = spark.read.csv(f"{os.environ.get('UNITY_CATALOG_VOLUME_PATH')}{os.environ.get('DATASET_DOWNLOAD_FILENAME')}", header=True, inferSchema=True)
  df_renamed_column = df.withColumnRenamed("First Name", "First_Name")
  return df_renamed_column

# COMMAND ----------

# MAGIC %md
# MAGIC Clean and prepare data

# COMMAND ----------

@dlt.table(
  comment="Data cleaned and prepared for analysis"
)
@dlt.expect("valid_first_name", "First_Name IS NOT NULL")
@dlt.expect_or_fail("valid_count", "Count > 0")
def dlt_silver():
  return (
    dlt.read("dlt_bronze")
    .withColumnRenamed("Year", "Year_Of_Birth")
    .select("Year_Of_Birth", "First_Name", "Count")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Aggregate data

# COMMAND ----------

@dlt.table(
  comment="A table summarizing counts of the top names for NY in 2021."
)
def dlt_gold():
  return (
    dlt.read("dlt_silver")
      .filter(expr("Year_Of_Birth == 2021"))
      .groupBy("First_Name")
      .agg(sum("Count").alias("Total_Count"))
      .sort(desc("Total_Count"))
  )