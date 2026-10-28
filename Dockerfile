
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=120 --retries=10 Flask==3.1.3 mlflow==3.17.2 pandas==3.1.0 scikit-learn==1.9.1 skops==0.16.0

COPY app.py .
COPY templates ./templates
COPY employee_attrition.csv .
COPY model ./model

EXPOSE 5000

CMD ["python", "app.py"]
