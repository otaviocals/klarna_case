from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.operators.athena import AWSAthenaOperator

import boto3

args = {
    "owner": "airflow",
}

with DAG(
    dag_id="test_dag_aws",
    default_args=args,
    schedule_interval="0 0 * * *",
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=60),
    tags=["example", "example2"],
    params={"example_key": "example_value"},
) as dag:

    test_athena_operator = AWSAthenaOperator(
        task_id=f"run_query_test",
        query="select * from credit_train_data where has_paid=TRUE",
        output_location="s3://klarna-case-model-bucket/credit-model/train/raw-train-data/",
        database="klarna_case",
    )


test_athena_operator

if __name__ == "__main__":
    dag.cli()
