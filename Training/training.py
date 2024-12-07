import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def trained_model(training_dataset_path, model_output_path):
    spark = SparkSession.builder \
    .appName("WineQualityModel") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.session.token", AWS_SESSION_TOKEN) \
    .getOrCreate()

    training_df = spark.read.csv(training_dataset_path, header=True, inferSchema=True)

    col_feature = training_df.columns[:-1] 
    vector_assembler = VectorAssembler(inputCols=col_feature, outputCol="features")
    training_df = vector_assembler.transform(training_df)

    rf = RandomForestClassifier(labelCol='quality', featuresCol='features')

    param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .build()

    model_evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")

    cross_validator = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=model_evaluator, numFolds=5)

    model = cross_validator.fit(training_df)

    model.bestModel.write().overwrite().save(model_output_path)

    spark.stop()

if __name__ == "__main__":
    pathto_training_dataset = "s3a://awsdatasetsbucket/CleanedTrainingDataSet.csv"
    pathto_model_output = "s3a://awsdatasetsbucket/trained/predict_model"

    trained_model(pathto_training_dataset, pathto_model_output)
