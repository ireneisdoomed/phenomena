
from langchain.embeddings.openai import OpenAIEmbeddings
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pinecone
import time
import itertools
from dotenv import load_dotenv
import os 
import nomic
from nomic import atlas
import numpy as np

load_dotenv()


def create_and_load_index():
    pinecone_api_key = os.getenv("PINECONE_TOKEN")
    pinecone.init(api_key=pinecone_api_key, environment="us-west4-gcp")
    pinecone.create_index("phenomena", dimension=1536)
    return pinecone.Index("phenomena")

def get_embeddings(texts, batch_size = 500):
    texts_chunks = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        texts_chunks.append(chunk)
    
    embeddings_list = []
    for i, batch in enumerate(texts_chunks):
        print(i)
        embeddings_list.append(embeddings.embed_documents(batch, chunk_size=len(batch)))
        time.sleep(1.2)
    return embeddings_list

def chunks(iterable, batch_size):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def insert_data_vector_store(data, index):
    for ids_vectors_chunk in chunks(data, batch_size=100):
        index.upsert(vectors=ids_vectors_chunk)

def create_atlas_mapping(ids,embeddings,metadata):
    nomic_api_key = os.getenv("NOMIC_TOKEN")
    nomic.login(nomic_api_key)
    data=[{'id': id, 'isPhenotype':meta['isPhenotype'],'isDisease':meta['isDisease'],} for id, meta in zip(ids, metadata)]

    atlas.map_embeddings(embeddings=embeddings,
                        data=data,
                        id_field='id',
                        colorable_fields=['isPhenotype','isDisease']
                        )

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("gs://ot-team/irene/matches_diseases")
grouped_df = (
    df
    .groupBy("efo_id", "isDisease", "isPhenotype").agg(f.collect_set(f.lower("label")).alias("labels"))
    .withColumn("isDisease", f.when(f.col("isDisease") == True, f.lit(1)).otherwise(f.lit(0))).withColumn("isPhenotype", f.when(f.col("isPhenotype") == True, f.lit(1)).otherwise(f.lit(0)))
    .withColumn("labels", f.array_join(f.col("labels"), ";"))
    .filter(f.length("labels") < 8_000)
)
grouped_pdf = grouped_df.toPandas()
texts = grouped_pdf["labels"].to_list()
ids = grouped_pdf["efo_id"].to_list()
metadata = grouped_pdf.drop(["efo_id", "labels"], axis=1).to_dict(orient="records")

# process embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_TOKEN"), model="text-embedding-ada-002")
embeddings_list = get_embeddings(texts)
embeddings_list_flat = [text for batch in embeddings_list for text in batch]

data = list(zip(ids, embeddings_list_flat, metadata))


index = create_and_load_index()
insert_data_vector_store(data, index)
create_atlas_mapping(ids, np.array(embeddings_list_flat), metadata)

# test similarity with diabetes
for i, e in enumerate(data):
    if e[0] == "EFO_0000400":
        vector = e[1]
        res = index.query(
            vector=vector,
            top_k=10,
            filter={"isDisease": 0, "isPhenotype": 1},
        )
        for e in res["matches"]:
            print(e["id"])
            print(e["score"])



# export results to tsv to visualise in https://projector.tensorflow.org/

import pandas as pd
import csv

metadata_df = grouped_pdf[["efo_id", "isDisease", "isPhenotype"]]
metadata_df.to_csv("gs://ot-team/irene/metadata.tsv", sep="\t", index=False)

with open('gs://ot-team/irene/vectors.tsv', 'w', newline='') as f_output:
     for row in data_df[["vector"]].iterrows():
         vector = row[1]["vector"]
         tsv_output = csv.writer(f_output, delimiter='\t')
         tsv_output.writerow(vector)



