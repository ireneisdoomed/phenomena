from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Create Disease Table").getOrCreate()

my_data = spark.read.parquet("gs://ot-team/irene/matches_diseases")
diseases = spark.read.parquet("gs://open-targets-pre-data-releases/23.02/output/etl/parquet/diseases")

mappings = diseases.selectExpr("id as efo_id", "name").join(my_data.select("efo_id").distinct(), on="efo_id")
mappings.write.csv("gs://ot-team/irene/matches_diseases_name_mappings", header=True, sep="\t")