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
        output_location="s3://klarna-case-model-bucket/credit-model/train/raw-train-data/{{ ds_nodash }}",
        database="klarna_case",
    )

    MODEL_NAME = "credit-model"
    BUCKET_NAME = "klarna-case-bucket"
    SM_ROLE = "klarna-case-sm-role"
    TRAIN_SCRIPT = "test.py"
    REGION = "us-east-1"
    QUERY_DATA_TASK_ID = "run_query_test"

    train_op = SageMakerTrainingOperator(
        task_id="test_training",
        config={
            "AlgorithmSpecification": {
                "TrainingImage": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
                "TrainingInputMode": "File",
            },
            "OutputDataConfig": {
                "S3OutputPath": "s3://"
                + BUCKET_NAME
                + "/"
                + MODEL_NAME
                + "/train/output-data/{{ ds_nodash }}"
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.c4.xlarge",
                "VolumeSizeInGB": 30,
            },
            "RoleArn": "" + SM_ROLE,
            "InputDataConfig": [
                {
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://"
                            + BUCKET_NAME
                            + "/"
                            + MODEL_NAME
                            + "/train/raw-train-data/{{ ds_nodash }}/{{ ti.xcom_pull(task_ids='run_query_test', key='return_value') }}.csv",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ChannelName": "training",
                }
            ],
            "HyperParameters": {
                "sagemaker_submit_directory": '"s3://'
                + BUCKET_NAME
                + "/"
                + MODEL_NAME
                + "/code/"
                + MODEL_NAME
                + '-1.0.tar.gz"',
                "sagemaker_program": MODEL_NAME
                + "-1.0/"
                + MODEL_NAME
                + "/"
                + TRAIN_SCRIPT,
                "sagemaker_container_log_level": "20",
                "sagemaker_job_name": MODEL_NAME + "-{{ ts_nodash }}",
                "sagemaker_region": REGION,
            },
            "TrainingJobName": MODEL_NAME + "-{{ ts_nodash }}",
        },
        wait_for_completion=True,
        dag=dag,
    )


test_athena_operator >> train_op

if __name__ == "__main__":
    dag.cli()
