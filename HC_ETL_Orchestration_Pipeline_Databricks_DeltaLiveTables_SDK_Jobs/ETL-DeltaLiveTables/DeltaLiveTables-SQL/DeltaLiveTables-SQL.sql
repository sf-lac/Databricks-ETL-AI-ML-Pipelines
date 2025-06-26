-- Databricks notebook source
-- MAGIC %md
-- MAGIC Ingest data into bronze (raw) table

-- COMMAND ----------

CREATE OR REFRESH LIVE TABLE dlt_sql_bronze
COMMENT "Data ingested from NY State Department of Health"
AS SELECT Year, `First Name` AS First_Name, County, Sex, Count 
FROM read_files(
  '/Volumes/main/default/unity_volume/rows.csv',
  format => 'csv',
  header => true,
  mode => 'FAILFAST'
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Clean and prepare data

-- COMMAND ----------

CREATE OR REFRESH LIVE TABLE dlt_sql_silver(
  CONSTRAINT valid_first_name EXPECT (First_Name IS NOT NULL),
  CONSTRAINT valid_count EXPECT (Count > 0) ON VIOLATION FAIL UPDATE
)
COMMENT "Data cleaned and prepared for analysis"
AS SELECT
  Year AS Year_Of_Birth,
  First_Name,
  Count
FROM live.dlt_sql_bronze;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Aggregate data

-- COMMAND ----------

CREATE OR REFRESH LIVE TABLE dlt_sql_gold
COMMENT "A table summarizing counts of the top names for NY in 2021."
AS SELECT
First_Name,
SUM(Count) AS Total_Count
FROM live.dlt_sql_silver
WHERE Year_Of_Birth = 2021
GROUP BY First_Name
ORDER BY Total_Count DESC;