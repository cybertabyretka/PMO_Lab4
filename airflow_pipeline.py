from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from download import download_dataset, preprocess_dataset
from train_model import train


dag = DAG(
    dag_id="PMO_Lab4_DAG",
    start_date=datetime(2025, 3, 8),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(
    task_id="download_dataset",
    python_callable=download_dataset,
    dag=dag
)

clear_task = PythonOperator(
    task_id = "clear_dataset",
    python_callable=preprocess_dataset,
    dag=dag
)

train_task = PythonOperator(
    python_callable=train,
    task_id = "train_model",
    dag=dag
)

download_task >> clear_task >> train_task