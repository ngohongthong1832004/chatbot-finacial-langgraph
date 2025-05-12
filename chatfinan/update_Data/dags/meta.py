# import json
# import logging
import pendulum
# import psycopg2
from airflow import DAG
# from pytz import timezone
# from confluent_kafka import Producer
# from airflow.decorators import task
from datetime import datetime, timedelta
# from confluent_kafka.admin import AdminClient, NewTopic, NewPartitions
# from logger import get_logger
# from datetime import datetime
# from dotenv import load_dotenv
# load_dotenv()
# import os

# POSTGRES_HOST = os.getenv("POSTGRES_HOST")
# POSTGRES_PORT = os.getenv("POSTGRES_PORT")
# POSTGRES_USER = os.getenv("POSTGRES_USER")
# POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
# POSTGRES_DB = os.getenv("POSTGRES_DB")


# logger = get_logger(logs_dir='/var/log/mention_schedule/', log_filename='update_mention_daily.log')

default_args = {
    'owner': 'social',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_keyword_to_kafka',
    default_args=default_args,
    description='Chạy một lần mỗi ngày lúc 2h sáng để lấy keywords từ PostgreSQL và gửi vào Kafka',
    schedule_interval='0 */8 * * *',
    start_date=datetime(2025, 1, 3, tzinfo=pendulum.timezone("Asia/Ho_Chi_Minh")),
    catchup=False,
) as dag:
    
#     @task
#     def fetch_keywords():
#         try:
#             connection = psycopg2.connect(
#                 user=POSTGRES_USER,
#                 password=POSTGRES_PASSWORD,
#                 host=POSTGRES_HOST,
#                 port=POSTGRES_PORT,
#                 database=POSTGRES_DB
#             )
#             cursor = connection.cursor()
#             logger.info("🟢 Kết nối đến PostgreSQL thành công")
#             cursor.execute("SELECT name FROM dim_keywords WHERE code IS NOT NULL and is_active = TRUE")
#             records = cursor.fetchall() 
#             keywords = [record[0] for record in records]
#             logger.info(f"🟢 Đã lấy {len(keywords)} từ khóa từ PostgreSQL")
#             cursor.close()
#             connection.close()
#             logger.info("❎ Đóng kết nối PostgreSQL")
#             return keywords
#         except Exception as e:
#             logger.error(f"🔴 Lỗi khi lấy từ khóa từ PostgreSQL: {e}")
#             raise

#     @task
#     def send_to_kafka(keywords):
#         """
#         Gửi danh sách từ khóa vào Kafka topic.
#         """
#         def delivery_callback(err, msg):
#             if err:
#                 logger.error(f"🔴 Lỗi gửi message: {err}")
#             else:
#                 logger.info(f"🟢 Message gửi thành công: {msg.topic()} [{msg.partition()}]")

#         try:
#             keyword_broker_update = 'keyword-broker-3:9093,keyword-broker-4:9093'
#             keyword_broker_new = 'keyword-broker-1:9093,keyword-broker-2:9093'
#             producer = Producer({
#                 'bootstrap.servers': keyword_broker_update,
#                 'client.id': 'airflow-producer',
#                 'retries': 5
#             })
#             logger.info("🟢 Kết nối Kafka thành công")
#             platforms = ['website', 'tiktok', 'youtube', 'facebook']

#             logger.info(f"📡 Đang gửi với platform: {platforms}")
#             for keyword in keywords:
#                 message = {
#                     'keyword': keyword,
#                     'params': 'day', 
#                     'timestamp': datetime.now(tz=timezone("Asia/Ho_Chi_Minh")).strftime('%Y-%m-%d %H:%M:%S'),
#                     'platforms': platforms
#                 }
#                 producer.produce('keyword-update', value=json.dumps(message), callback=delivery_callback)

#             producer.flush()
#             logger.info("✅ Hoàn tất gửi tất cả từ khóa lên Kafka")
#         except Exception as e:
#             logger.error(f"🔴 Lỗi khi gửi message đến Kafka: {e}")
#         finally:
#             logger.info("❎ Đóng kết nối Kafka")

#     keywords = fetch_keywords()
#     send_to_kafka(keywords)  


import yfinance as yf
import pandas as pd

ticker_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'NVDA', 'NFLX', 'SPY', 'AMD',
                  'BABA', 'INTC', 'GOOG', 'V', 'DIS']

stock_data = yf.download(ticker_symbols, period="1mo", interval="1d")
stock_data_cleaned = stock_data['Close'].dropna(axis=1, how='all')
stock_data_cleaned.columns = [col for col in stock_data_cleaned.columns]

stock_data_df = pd.DataFrame(stock_data_cleaned)
stock_data_df.head(10)