# Matala 4 MN
# Erga 207829813
# Neta Cohen 325195774

from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("BooksApp").getOrCreate()

# Read the JSON file
df = spark.read.json("books.json")

# Filter books with authors whose first name starts with "F"
filtered_df = df.filter(df.author.startswith("F"))

# Calculate the number of years passed since the book was printed
result_df = filtered_df.withColumn("years_passed", 2023 - df.year)

# Select the required columns and print the result
result_df.select("title", "author", "years_passed").show(truncate=False)
