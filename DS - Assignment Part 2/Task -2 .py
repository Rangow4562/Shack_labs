#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries
import pyspark
from pyspark.sql import SparkSession, functions as F

sc = pyspark.SparkContext('local[*]')
spark = SparkSession     .builder     .getOrCreate()


# # Loading Data

# ### Reading the AMZ data

# In[12]:


amazon_ddf = (spark.read.csv('amz_com-ecommerce_sample.csv', header=True, inferSchema=True).withColumnRenamed("product_name","Amazon_Product").withColumnRenamed("retail_price","A_retail_price").withColumnRenamed("discounted_price","A_discounted_price"))
amazon_ddf = amazon_ddf.select('ID','uniq_id','Amazon_Product', 'A_retail_price', 'A_discounted_price')
amazon_ddf.show(5, False)
print(f"{amazon_ddf.count()} products")
amazon_ddf.printSchema()


# ### Reading the flipkart data

# In[3]:


flipkart_ddf = (spark.read.csv('flipkart_com-ecommerce_sample.csv', header=True, inferSchema=True).withColumnRenamed("product_name","Flipkart_Product").withColumnRenamed("retail_price","F_retail_price").withColumnRenamed("discounted_price","F_discounted_price"))
flipkart_ddf = flipkart_ddf.select('ID','uniq_id','Flipkart_Product', 'F_retail_price', 'F_discounted_price')
flipkart_ddf.show(5, False)
print(f"{flipkart_ddf.count()} products")
flipkart_ddf.printSchema()


# ## When joining on (exact) Product

# In[4]:


result = amazon_ddf.join(flipkart_ddf)
result.count()
result.show(1)


# ## Record Linkage (Fuzzy Matching)

# ### Prepare join column by doing multiple transformations

# In[5]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Tokenizer, NGram, HashingTF, MinHashLSH, RegexTokenizer, SQLTransformer

model = Pipeline(stages=[
    SQLTransformer(statement="SELECT *, lower(Flipkart_Product) lower FROM __THIS__"),
    Tokenizer(inputCol="lower", outputCol="token"),
    StopWordsRemover(inputCol="token", outputCol="stop"),
    SQLTransformer(statement="SELECT *, concat_ws(' ', stop) concat FROM __THIS__"),
    RegexTokenizer(pattern="", inputCol="concat", outputCol="char", minTokenLength=1),
    NGram(n=2, inputCol="char", outputCol="ngram"),
    HashingTF(inputCol="ngram", outputCol="vector"),
    MinHashLSH(inputCol="vector", outputCol="lsh", numHashTables=3)
]).fit(flipkart_ddf)

result_fpk = model.transform(flipkart_ddf)
result_fpk = result_fpk.filter(F.size(F.col("ngram")) > 0)
# print(f"{result_lens.count()} products:")
result_lens.select('ID', 'Flipkart_Product', 'concat', 'char', 'ngram', 'vector', 'lsh').show(1)


# #### Since the columns have different names, we need to define which columns to match.Behind the scenes, fuzzymatcher determines the best match for each combination. For this data set we are analyzing over 14 million combinations.

# In[6]:


# Use pipeline previous defined
model2 = Pipeline(stages=[
    SQLTransformer(statement="SELECT *, lower(Amazon_Product) lower FROM __THIS__"),
    Tokenizer(inputCol="lower", outputCol="token"),
    StopWordsRemover(inputCol="token", outputCol="stop"),
    SQLTransformer(statement="SELECT *, concat_ws(' ', stop) concat FROM __THIS__"),
    RegexTokenizer(pattern="", inputCol="concat", outputCol="char", minTokenLength=1),
    NGram(n=2, inputCol="char", outputCol="ngram"),
    HashingTF(inputCol="ngram", outputCol="vector"),
    MinHashLSH(inputCol="vector", outputCol="lsh", numHashTables=3)
]).fit(amazon_ddf)

result_amz = model2.transform(amazon_ddf)
result_amz = result_amz.filter(F.size(F.col("ngram")) > 0)
# print(f"Example transformation ({result_imdb.count()} movies left):")
result_amz.select('ID', 'Amazon_Product', 'concat', 'char', 'ngram', 'vector', 'lsh').show(1)


# ### Join based on Jaccard Distance

# In[7]:


result = model.stages[-1].approxSimilarityJoin(result_amz, result_fpk, 0.5, "jaccardDist")
# print(f"{result.count()} matches")
(result
 .select('datasetA.ID', 'datasetA.Amazon_Product', 'datasetB.Flipkart_Product', 'jaccardDist')
 .sort(F.col('datasetA.ID')))


# ### Optimization: Only keep single row with lowest distance

# In[8]:


from pyspark.sql import Window
w = Window.partitionBy('datasetA.ID')
result = (result
           .withColumn('minDist', F.min('jaccardDist').over(w))
           .where(F.col('jaccardDist') == F.col('minDist'))
           .drop('minDist'))
(result
 .select('datasetA.Amazon_Product', 'datasetB.Flipkart_Product', 'jaccardDist')
 .sort(F.col('datasetA.ID')).show)


# ## Final DataFrame

# In[13]:


df = result.select('datasetA.Amazon_Product', 'datasetA.A_retail_price','datasetA.A_discounted_price','datasetB.Flipkart_Product', 'datasetB.F_retail_price', 'datasetB.F_discounted_price')


# This inconsistency features the need to ensure you truly comprehend your information and what cleaning and separating you might have to do prior to attempting to coordinate.
# 
# We've taken a gander at the outrageous cases, we should investigate a portion of the matches that may be somewhat more testing by taking a gander at scores < 80:

# In[14]:


df


# In[15]:


def toCSV(spark_df, n=None, save_csv=None, csv_sep=',', csv_quote='"'):
        """get spark_df from hadoop and save to a csv file

        Parameters
        ----------
        spark_df: incoming dataframe
        n: number of rows to get
        save_csv=None: filename for exported csv

        Returns
        -------

        """

        # use the more robust method
        # set temp names
        tmpfilename = save_csv or (wfu.random_filename() + '.csv')
        tmpfoldername = wfu.random_filename()
        print (n)
        # write sparkdf to hadoop, get n rows if specified
        if n:
            spark_df.limit(n).write.csv(tmpfoldername, sep=csv_sep, quote=csv_quote)
        else:
            spark_df.write.csv(tmpfoldername, sep=csv_sep, quote=csv_quote)

        # get merge file from hadoop
        HDFSUtil.getmerge(tmpfoldername, tmpfilename)
        HDFSUtil.rmdir(tmpfoldername)

        # read into pandas df, remove tmp csv file
        pd_df = pd.read_csv(tmpfilename, names=spark_df.columns, sep=csv_sep, quotechar=csv_quote)
        os.remove(tmpfilename)

        # re-write the csv file with header!
        if save_csv is not None:
            df.to_csv(save_csv, sep=csv_sep, quotechar=csv_quote)


# In[ ]:




