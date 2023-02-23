# Databricks notebook source
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType

custfile = 'dbfs:///FileStore/tables/internet_service_churn.csv'

custschema = StructType([StructField('id', LongType(), False), 
                         StructField('is_tv_subscriber', LongType(), True),
                         StructField('is_movie_package_subscriber', LongType(), True),
                         StructField('subscription_age', DoubleType(), True),
                         StructField('bill_avg', LongType(), True),
                         StructField('remaining_contract', DoubleType(), True),
                         StructField('service_failure_count', LongType(), True),
                         StructField('download_avg', DoubleType(), True),
                         StructField('upload_avg', DoubleType(), True),
                         StructField('download_over_limit', LongType(), True),
                         StructField('churn', LongType(), True)])

custdf = spark.read.format('csv').option('header', True).schema(custschema).load(custfile)
custdf.show()

# COMMAND ----------

# We dony need the ID so we will drop it
custdf = custdf.drop("id")

#Rename Churn with label
custdf = custdf.withColumnRenamed("churn","label")

#Separating the columns in categorical and continuous
cat_cols = ['is_tv_subscriber','is_movie_package_subscriber']
con_cols = ['subscription_age','bill_avg','remaining_contract','service_failure_count','download_avg','upload_avg','download_over_limit']
target_col = ['label']
print("The categorial cols are : ", cat_cols)
print("The continuous cols are : ", con_cols)
print("The target variable is :  ", target_col)

# COMMAND ----------

from pyspark.sql.functions import isnull
from pyspark.sql.functions import *

#Check missing data

total_missing =  custdf.select([count(when(col(c).isNull(),c)).alias(c) for c in custdf.columns])

display(total_missing)

# COMMAND ----------

#Check the data for null download average
custdf.filter(col("download_avg").isNull()).sample(withReplacement=False, fraction=0.20).show()
custdf.filter(col("upload_avg").isNull()).sample(withReplacement=False, fraction=0.20).show()

print('Total Number of rows:',custdf.count())
percentmiss = (381/custdf.count())*100
print('Percent of Missing value for download and upload:', percentmiss)

# COMMAND ----------

# MAGIC %md
# MAGIC According to the data information on remaining_contract,it is described as how many year remaining for customer contract. If null, that means customer hasnt have a contract. The customer who has a contract time have to use their service until contract end. if they canceled their service before contract time end they pay a penalty fare.
# MAGIC 
# MAGIC Which mean null data are customer doesnt have any contract, and don't have the obligation to pay penalty in case they cancelled the service.
# MAGIC In this case we can replace the null data with 0 instead.
# MAGIC 
# MAGIC Since the missing value for download and upload average are within the same rows, and adds up only 0.5% of the whole data, I'm just going to remove the missing rows.   

# COMMAND ----------

#Fill Null value with 0 for remaining contract
custdf = custdf.na.fill(value=0,subset=['remaining_contract'])

#Remove the missing null rows for upload and download average
custdf = custdf.na.drop(subset=['download_avg','upload_avg'])

total_missing =  custdf.select([count(when(col(c).isNull(),c)).alias(c) for c in custdf.columns])

display(total_missing)

# COMMAND ----------

# Describe continous column  data
custdf[con_cols].describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Theres a minus value for subscription age which is unusual, since its a value that describe how long a customer has used the service. 
# MAGIC There's a very huge gap in bills between customer, while the average is only 19, the highest one is 406, same with download_avg. 

# COMMAND ----------

#Check the distribution using databricks Visualization
display(custdf[con_cols])

# COMMAND ----------

# Check for outliers in subscription age
custdf.orderBy(asc("subscription_age")).limit(5).show()

# Remove the minus value from subscription age
custdf = custdf[custdf["subscription_age"] >= 0]

# COMMAND ----------

# Check for outliers billing average
custdf.orderBy(desc("bill_avg")).limit(20).show()

# COMMAND ----------

# Check for outliers download average
custdf.orderBy(desc("download_avg")).limit(20).show()

# COMMAND ----------

# There are customer churn which has values 0 for all columns
# Probably Data of past churned customer
# Drop this data as it doesnt reflect our current data
import pyspark.sql.functions as F

condition = (custdf.is_tv_subscriber == 0) & (custdf.is_movie_package_subscriber == 0) & (custdf.subscription_age == 0) & (custdf.bill_avg == 0) & (custdf.remaining_contract == 0) & (custdf.service_failure_count == 0) & (custdf.download_avg == 0.0) & (custdf.upload_avg == 0.0) & (custdf.download_over_limit == 0)


custdf = custdf.filter(~condition)

# COMMAND ----------

#Correlation Analysis

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=custdf.columns, outputCol=vector_col)
df_vector = assembler.transform(custdf).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)

cor_np = matrix.collect()[0][matrix.columns[0]].toArray()
cor_np

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Remaining time seems to has the most negative correlation towards customer churn, It seems customers are not willing to stop the service if they have to pay the penalty.

# COMMAND ----------

# Split Data

# Split the data into 70% training and 30% testing
traindf, testdf = custdf.randomSplit(weights=[0.8,0.2], seed=200)

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, Bucketizer, Binarizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

#Bucketizer for Subscription Age
agesplits = [0,2,4,6,8,float("inf")]
SubsAgeBucketizer = Bucketizer(splits=agesplits, inputCol="subscription_age", outputCol="SubsAgeBucket")

# Train and Fit
trainingData = SubsAgeBucketizer.transform(traindf)

#Create Feature
feature = VectorAssembler(inputCols=['is_tv_subscriber','is_movie_package_subscriber','SubsAgeBucket','bill_avg','remaining_contract','service_failure_count','download_avg','upload_avg','download_over_limit'], outputCol = 'features')
trainingData = feature.transform(trainingData)

#Standard Scaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Standard Scaler the features
scalerModel = scaler.fit(trainingData)
trainingData = scalerModel.transform(trainingData)


# Set Regression Model
# Logistic Regression
lr = LogisticRegression(maxIter = 10, regParam = 0.01,  featuresCol = 'scaledFeatures')

trainingData.show()

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [SubsAgeBucketizer, feature, scaler, lr]

p = Pipeline(stages=stages)

pModel = p.fit(traindf)
# Compare the model with original training data
check = pModel.transform(traindf).select('probability','label','rawPrediction','prediction')

#See the result on test data
pred = pModel.transform(testdf).select('probability','label','rawPrediction','prediction')
# Evaluate
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print("Evaluation with test data result:", evaluator.evaluate(pred))

print(" \n\nPrediction for the test data")
pred.show()

# COMMAND ----------

#Test using Decision Tree
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

dt = DecisionTreeClassifier(featuresCol="features")

stages = [SubsAgeBucketizer, feature, dt]

p = Pipeline(stages=stages)

model = p.fit(traindf)

predictions = model.transform(testdf)


predictions.select("prediction", "label", "features").show()

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Evaluation with test data result:", accuracy)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]
print(treeModel)
print('Feature Importance:',treeModel.featureImportances)

# COMMAND ----------

#Test with Random Forest 
from pyspark.ml.classification import RandomForestClassifier

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=9)
stages = [SubsAgeBucketizer, feature, rf]

p = Pipeline(stages=stages)
model = p.fit(traindf)

# Make predictions.
predictions = model.transform(testdf)

#predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Evaluation with test data result:", accuracy)
print("Test Error = %g" % (1.0 - accuracy))
rfModel = model.stages[2]
print(rfModel)  # summary 
print('Feature Importance:',rfModel.featureImportances)

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

# Train a GBT model.
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

stages = [SubsAgeBucketizer, feature, gbt]

p = Pipeline(stages=stages)
model = p.fit(traindf)

# Make predictions.
predictions = model.transform(testdf)

#predictions.select("prediction", "label", "features").show(5)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Evaluation with test data result:", accuracy)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)
print('Feature Importance:',gbtModel.featureImportances)
