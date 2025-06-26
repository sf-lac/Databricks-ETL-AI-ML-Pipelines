# Databricks notebook source
# MAGIC %md
# MAGIC Install Databricks SDK

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade

# COMMAND ----------

# MAGIC %pip show databricks-sdk | grep -oP '(?<=Version: )\S+'

# COMMAND ----------

# MAGIC %md
# MAGIC Create Databricks job

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, PipelineTask, Source

w = WorkspaceClient()

job_name = "Databricks SDK Job"
description = "Databricks SDK Job to Run a Delta Live Tables Data Pipeline"
pipeline_id = "ce45eeba-9337-4580-9b4a-402ac3ae231f"
task_key = "dlt_pipeline_task"

print("Attempting to create the job. Please wait...\n")

j = w.jobs.create(
    name = job_name,
    tasks = [
        Task(
            description = description,           
            pipeline_task = PipelineTask(
                pipeline_id=pipeline_id,
                full_refresh=False                
            ),
            task_key = task_key
        )
    ]
)

print(f"View the job at {w.config.host}/#job/{j.job_id}\n")

# COMMAND ----------

run_now_response = w.jobs.run_now(job_id=j.job_id)
run_now_response.response.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC Created new Delta Live Tables pipeline update with ID: 841f563c-7fba-460b-939b-282ee46ca592 from existing one, ce45eeba-9337-4580-9b4a-402ac3ae231f

# COMMAND ----------

# MAGIC %md
# MAGIC Cancel job

# COMMAND ----------

cancelled_run = w.jobs.cancel_run(run_id=run_now_response.response.run_id).result()
cancelled_run

# COMMAND ----------

# MAGIC %md
# MAGIC Delete job

# COMMAND ----------

w.jobs.delete(job_id=j.job_id)

# COMMAND ----------

display(spark.sql("SELECT * FROM main.`dlt-sql`.dlt_sql_bronze limit 10"))

# COMMAND ----------

display(spark.sql("SELECT * FROM main.`dlt-sql`.dlt_sql_silver limit 10"))

# COMMAND ----------

display(spark.sql("SELECT * FROM main.`dlt-sql`.dlt_sql_gold limit 10"))