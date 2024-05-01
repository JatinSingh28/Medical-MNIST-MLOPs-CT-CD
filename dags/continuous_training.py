from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


def extract():
    return 0

def train():
    return 0


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

run_extract = PythonOperator(task_id = "extract", python_callable=extract, dag=dag)

run_train = PythonOperator(task_id = "train", python_callable=train, dag=dag)

run_extract >> run_train