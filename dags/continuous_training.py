from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from train import continious_train
import os

sys.path.append("../")
from aws_controls import AwsControl
from config import config

aws = AwsControl(config["aws_key"], config["aws_secret"])


def download_aws_data():
    aws.download_s3_bucket(local_dir="AWS_data")


def train():
    continious_train()


def delete_aws_data():
    aws.delete_s3_folder_images(local_dir="AWS_data")
    
def delete_downloaded_aws_data():
    base_path = "./AWS_data"
    dirs = os.listdir(base_path)
    for folder in dirs:
        files = os.listdir(os.path.join(base_path,folder))
        for file_name in files:
            os.remove(os.path.join(base_path,folder,file_name))


default_args = {
    "owner": "Jatin",
    "depends_on_past": False,
    "start_date": datetime(2024, 4, 29),
    "email": ["sagoisinghjatin9951@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "Medical-Image-CT",
    default_args=default_args,
    description="Continuous Training",
    schedule="@weekly",
)

# run_extract = PythonOperator(task_id="extract", python_callable=extract, dag=dag)
run_download = PythonOperator(
    task_id="download_data", python_callable=download_aws_data, dag=dag
)
run_train = PythonOperator(task_id="train", python_callable=train, dag=dag)
run_delete = PythonOperator(task_id="delete", python_callable=delete_aws_data, dag=dag)
run_local_delete = PythonOperator(task_id="delete_local", python_callable=delete_downloaded_aws_data, dag=dag)

run_download >> run_train >> run_delete >> run_local_delete
