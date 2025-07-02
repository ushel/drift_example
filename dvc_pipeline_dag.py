from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'utkarsh',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG('dvc_pipeline',
         default_args=default_args,
         schedule_interval='@daily',  # or '@hourly', etc.
         catchup=False) as dag:

    detect_drift = BashOperator(
        task_id='detect_drift',
        bash_command='dvc repro detect_drift'
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='dvc repro train_model'
    )

    detect_drift >> train_model
