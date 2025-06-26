-- Databricks notebook source
-- MAGIC %md
-- MAGIC Create a managed Delta table

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS cord_raw;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Ingest data from AWS S3 cloud storage

-- COMMAND ----------

COPY INTO cord_raw
FROM (SELECT * FROM 's3://ai2-semanticscholar-cord-19/2020-05-26/metadata.csv')
FILEFORMAT = CSV
FORMAT_OPTIONS ('inferSchema'='true', 'header'='true')
COPY_OPTIONS ('mergeSchema'='true');


-- COMMAND ----------

DESCRIBE cord_raw;

-- COMMAND ----------

SELECT count(*)  FROM cord_raw LIMIT 5;

-- COMMAND ----------

SELECT * FROM cord_raw LIMIT 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Create a shallow clone

-- COMMAND ----------

CREATE OR REPLACE TABLE cord_clean SHALLOW CLONE cord_raw;

-- COMMAND ----------

DESCRIBE cord_clean;

-- COMMAND ----------

SELECT count(*) FROM cord_clean;

-- COMMAND ----------

INSERT INTO cord_raw (cord_uid) VALUES ('xxxxxxx'); 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Bring new data into the clone for further cleaning and processing

-- COMMAND ----------

MERGE INTO cord_clean
USING cord_raw
ON cord_raw.cord_uid = cord_clean.cord_uid
WHEN NOT MATCHED THEN INSERT *;
