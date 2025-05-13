FROM python:3.11-slim

WORKDIR /app

COPY . . 

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "run.py"]

