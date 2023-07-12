# Matala 4 MN
# Erga 207829813
# Neta Cohen 325195774

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum

# Create a SparkSession
spark = SparkSession.builder.appName("BooksApp").getOrCreate()

# Read the JSON file
df = spark.read.json("books.json")

# Filter books written in English
english_books = df.filter(df.language == "English")

# Group by author and calculate the total number of pages
pages_per_author = english_books.groupBy("author").agg(sum("pages").alias("total_pages"))

# Print the results
pages_per_author.show(truncate=False)
