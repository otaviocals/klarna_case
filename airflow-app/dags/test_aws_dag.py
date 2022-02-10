from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.operators.athena import AWSAthenaOperator
from airflow.contrib.operators.sagemaker_training_operator import (
    SageMakerTrainingOperator,
)

import boto3
from sagemaker.workflow.airflow import transform_config_from_estimator
from sagemaker.sklearn.estimator import SKLearn

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
        output_location="s3://klarna-case-model-bucket/credit-model/train/raw-train-data/raw-train-data.csv",
        database="klarna_case",
    )

    train_config = transform_config_from_estimator(
        estimator=SKLearn(
            entry_point="test.py",
            source_dir="s3://klarna-case-model-bucket/credit-model/code/credit-model.tar.gz",
            # role=role,
            framework_version="0.23-1",
            instance_type="ml.c4.xlarge",
        ),
        task_id="tf_training",
        task_type="training",
        instance_count=1,
        instance_type="ml.m4.xlarge",
        data="s3://klarna-case-model-bucket/credit-model/train/raw-train-data/raw-train-data.csv",
        content_type="text/csv",
    )

    train_op = SageMakerTrainingOperator(
        task_id="test_training", config=train_config, wait_for_completion=True, dag=dag
    )


test_athena_operator >> train_op

if __name__ == "__main__":
    dag.cli()
