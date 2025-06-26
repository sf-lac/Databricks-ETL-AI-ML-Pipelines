# Databricks notebook source
# MAGIC %md
# MAGIC #### Named Entity Recognition in Clinical Notes using spaCy and Med7
# MAGIC
# MAGIC Named Entity Recognition (`NER`) is a subtask of Natural Language Processing (`NLP`) that involves identifying and classifying named entities in unstructured data/text into predefined categories. 
# MAGIC
# MAGIC `spaCy` is library for advanced NLP in Python. It comes with pretrained pipelines and features neural network models for tagging, parsing, named entity recognition, text classification and more.
# MAGIC
# MAGIC `Med7` is a transferable clinical NLP processing model compatible with spaCy for `clinical` NER tasks. The `en_core_med7_lg` model is trained to recognize 7 categories: `DRUG`, `DOSAGE`, `STRENGTH`, `ROUTE`, `FREQUENCY`, `DURATION`, `FORM`, and is hosted on Huggingface.

# COMMAND ----------

# MAGIC %md
# MAGIC Install model

# COMMAND ----------

# MAGIC %pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC Load model

# COMMAND ----------

import spacy

med7 = spacy.load("en_core_med7_lg")

# create distinct colours for labels
col_dict = {}
seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
for label, colour in zip(med7.pipe_labels['ner'], seven_colours):
    col_dict[label] = colour

options = {'ents': med7.pipe_labels['ner'], 'colors':col_dict}

# COMMAND ----------

# MAGIC %md
# MAGIC Ingest data from AWS S3 into Databricks

# COMMAND ----------

import os
from pyspark.context import SparkContext 
from pyspark import SparkConf

# COMMAND ----------

data_path = '/FileStore/onco/data/'
os.environ['data_path'] = f'/dbfs{data_path}'
delta_path = '/mnt/onco/delta'

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p $data_path
# MAGIC cd $data_path
# MAGIC wget https://hls-eng-data-public.s3.amazonaws.com/data/mt_onc_50.zip -P $data_path
# MAGIC unzip -o mt_onc_50.zip

# COMMAND ----------

ls /dbfs/FileStore/onco/data/mt_onc_50/

# COMMAND ----------

dbutils.fs.cp("file:/dbfs/FileStore/onco/data/mt_onc_50/", "dbfs:/FileStore/onco", True)

# COMMAND ----------

sc = SparkContext.getOrCreate(SparkConf())
sdf = sc.wholeTextFiles("/FileStore/onco/").toDF().withColumnRenamed('_1', 'path').withColumnRenamed('_2', 'text')

# COMMAND ----------

display(sdf.limit(5))

# COMMAND ----------

sdf.limit(1).select("text").collect()[0]

# COMMAND ----------

# MAGIC %md
# MAGIC Write to delta bronze layer

# COMMAND ----------

sdf.write.format('delta').mode('overwrite').save(f'{delta_path}/bronze/onco')
display(dbutils.fs.ls(f'{delta_path}/bronze/onco'))

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * from delta.`/mnt/onco/delta/bronze/onco/`;

# COMMAND ----------

sample_text = sdf.limit(1).select("text").collect()[0]
sample_text.text

# COMMAND ----------

# MAGIC %md
# MAGIC Perform NER with Med7

# COMMAND ----------

text = sample_text.text
doc = med7(text)

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize NERs

# COMMAND ----------

spacy.displacy.render(doc, style='ent', jupyter=True, options=options)

# COMMAND ----------

import numpy as np
import pandas as pd

entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
entities_df = pd.DataFrame(entities, columns=['Chunk', 'Entity', 'Start', 'End'])
entities_df

# COMMAND ----------

import plotly.express as px

fig = px.scatter(entities_df, x="Start", y="End", color="Entity", hover_data=["Chunk"])
fig.update_layout(title="Named Entities in Clinical Note")
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Further processing

# COMMAND ----------

clinical_notes  = spark.sql("SELECT path, text from delta.`/mnt/onco/delta/bronze/onco/`").toPandas()

# COMMAND ----------

clinical_notes_nlp = {clinical_notes.path[i]:med7(clinical_notes.text[i]) for i in range(len(clinical_notes))}

# COMMAND ----------

#for i in range(len(clinical_notes_nlp)):
#    spacy.displacy.render(clinical_notes_nlp[f'dbfs:/FileStore/onco/mt_oncology_{i}.txt'], style='ent', jupyter= True,  options=options)

# COMMAND ----------

ner_pdf = pd.DataFrame()
for i in range(len(clinical_notes_nlp)):
    path = f"dbfs:/FileStore/onco/mt_oncology_{i}.txt"    
    named_entities = {e.text: e.label_ for e in clinical_notes_nlp[path].ents}    
    ner_pdf_i = pd.DataFrame(named_entities.items(), columns=['chunk', 'entity'])
    ner_pdf_i.insert(loc=0, column='path', value=path)    
    ner_pdf = pd.concat([ner_pdf, ner_pdf_i], axis=0)

# COMMAND ----------

ner_pdf.head(22)

# COMMAND ----------

from collections import Counter

drug_entities = dict([e for e in Counter(list(ner_pdf[ner_pdf.entity == "DRUG"]["chunk"])).most_common() if e[1] >= 6])
drug_entities_df = pd.DataFrame(drug_entities.items(), columns=['Drug', 'Frequency'])
drug_entities_df

# COMMAND ----------

px.bar(drug_entities_df, x='Drug', y = 'Frequency', title = "Frequency of drugs in clinical notes")

# COMMAND ----------

# MAGIC %md
# MAGIC #### RxNorm coding for drug entities extracted from clinical notes
# MAGIC
# MAGIC RxNorm is a standardized nomenclature for clinical drugs produced by the U.S. National Library of Medicine (NLM). RxNorm represents one of a suite of designated standards for use in U.S. Federal government systems for the electronic exchange of clinical health information.
# MAGIC
# MAGIC The goal of RxNorm is to allow various systems using different drug nomenclatures to share data efficiently.
# MAGIC
# MAGIC RxNorm provides standard names for clinical drugs (active ingredient + strength + dose form) and for dose forms as administered to a patient. It provides links from clinical drugs, both branded and generic, to their active ingredients, drug components (active ingredient + strength), and related brand names.

# COMMAND ----------

base_url = "https://rxnav.nlm.nih.gov/REST"
drug_ents = ["Coumadin", "Amiodarone", "Lovenox", "cisplatin"]
rxcuis = []

# COMMAND ----------

# MAGIC %pip install xmltodict

# COMMAND ----------

import requests
import json
import xmltodict
import numpy as np
import pandas as pd

# COMMAND ----------

for drug_ent in drug_ents:
    # Construct the API request URL
    request_url = f"{base_url}/drugs?name={''.join(drug_ent)}"   

    # Send the API request and retrieve the response
    response = requests.get(request_url)

    # Convert response XML to JSON
    response_json_dict = xmltodict.parse(response.content)
    response_json_str = json.dumps(response_json_dict['rxnormdata'])
    response_json = json.loads(response_json_str)

    # Extract the RxCUI and concept name    
    for group in response_json["drugGroup"]["conceptGroup"]:       
        if "conceptProperties" in group.keys():
            concept = group["conceptProperties"]   
            if isinstance(concept, dict):
                rxcui = group["conceptProperties"]["rxcui"]
                name = group["conceptProperties"]["name"] 
                rxcuis.append([drug_ent, rxcui, name])   
            elif isinstance(concept, list):
                for i in range(len(concept)):
                    rxcui = group["conceptProperties"][i]["rxcui"]
                    name = group["conceptProperties"][i]["name"]       
                    rxcuis.append([drug_ent, rxcui, name])
    
# Create Pandas DataFrame from RxNorm data
rxnorm_pdf = pd.DataFrame(rxcuis, columns=['Drug', 'RxCUI', 'Concept'])

# COMMAND ----------

rxnorm_pdf

# COMMAND ----------

drug_ent = "Lovenox"

# COMMAND ----------

# Construct the API request URL
request_url = f"{base_url}/drugs?name={''.join(drug_ent)}"


# COMMAND ----------

# Send the API request and retrieve the response
response = requests.get(request_url)

# COMMAND ----------

response.headers

# COMMAND ----------

response.content

# COMMAND ----------

response_json_dict = xmltodict.parse(response.content)
response_json_str = json.dumps(response_json_dict['rxnormdata'])
response_json = json.loads(response_json_str)

# COMMAND ----------

response_json

# COMMAND ----------

# Extract the RxCUI and concept name
rxcuis = []
for group in response_json["drugGroup"]["conceptGroup"]:       
    if "conceptProperties" in group.keys():
        concept = group["conceptProperties"]   
        if isinstance(concept, dict):
            rxcui = group["conceptProperties"]["rxcui"]
            name = group["conceptProperties"]["name"] 
            rxcuis.append([drug_ent, rxcui, name])   
        elif isinstance(concept, list):
            for i in range(len(concept)):
                rxcui = group["conceptProperties"][i]["rxcui"]
                name = group["conceptProperties"][i]["name"]       
                rxcuis.append([drug_ent, rxcui, name])


# COMMAND ----------

rxnorm_pdf_lovenox = pd.DataFrame(rxcuis, columns=['Drug', 'RxCUI', 'Concept'])

# COMMAND ----------

rxnorm_pdf_lovenox
