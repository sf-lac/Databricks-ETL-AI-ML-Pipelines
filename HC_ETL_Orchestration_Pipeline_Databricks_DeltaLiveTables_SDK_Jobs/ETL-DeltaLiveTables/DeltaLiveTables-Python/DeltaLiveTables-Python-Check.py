# Databricks notebook source
display(spark.sql("SELECT * FROM main.dlt_python.dlt_bronze limit 10"))

# COMMAND ----------

display(spark.sql("SELECT * FROM main.dlt_python.dlt_silver limit 10"))

# COMMAND ----------

display(spark.sql("SELECT * FROM main.dlt_python.dlt_gold limit 10"))