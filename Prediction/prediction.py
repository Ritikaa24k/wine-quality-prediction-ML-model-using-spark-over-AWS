import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def perform_prediction(
    trained_model_location,
    validation_data_source,
    input_feature_list
):
    # Establish a Spark session with S3 configurations
    spark_session = (
        SparkSession.builder
        .appName("WineQualityModelEvaluation")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
        .config("spark.hadoop.fs.s3a.session.token", os.environ["AWS_SESSION_TOKEN"])
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
        .getOrCreate()
    )

    # Load the pre-trained Random Forest Classification model
    random_forest_model = RandomForestClassificationModel.load(trained_model_location)

    # Read the validation set from the provided path
    validation_data = spark_session.read.csv(validation_data_source, header=True, inferSchema=True)

    # Assemble feature columns into a single vector column
    assembler = VectorAssembler(inputCols=input_feature_list, outputCol="features")
    assembled_validation_data = assembler.transform(validation_data)

    # Generate predictions
    predictions = random_forest_model.transform(assembled_validation_data)

    # Evaluate the model using the F1 score
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_metric = evaluator.evaluate(predictions)

    print("F1 Score:", f1_metric)

    spark_session.stop()

if __name__ == "__main__":
    model_s3_path = "s3a://awsdatasetsbucket/trained/predict_model"
    val_s3_path = "s3a://awsdatasetsbucket/Cleaned_ValidationDataset.csv"

    features = [
        "fixed acidity", "volatile acidity", "citric acid", "chlorides",
        "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
    ]

    perform_prediction(model_s3_path, val_s3_path, features)
