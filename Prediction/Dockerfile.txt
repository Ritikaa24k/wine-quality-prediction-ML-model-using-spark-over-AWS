FROM bitnami/spark:latest

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["spark-submit", "prediction.py"]
