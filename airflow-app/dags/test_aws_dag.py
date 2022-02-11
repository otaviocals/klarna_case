from datetime import timedelta
import yaml
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.operators.athena import AWSAthenaOperator
from airflow.contrib.operators.sagemaker_training_operator import (
    SageMakerTrainingOperator,
)
from airflow.contrib.operators.sagemaker_model_operator import SageMakerModelOperator
import boto3


def generate_sagemaker_train_config(
    MODEL_NAME, BUCKET_NAME, SM_ROLE, TRAIN_SCRIPT, REGION, QUERY_DATA_TASK_ID
):
    config = {
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
                        + "/train/raw-train-data/{{ ds_nodash }}/{{ ti.xcom_pull(task_ids='"
                        + QUERY_DATA_TASK_ID
                        + "', key='return_value') }}.csv",
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
            "sagemaker_program": MODEL_NAME + "-1.0/" + MODEL_NAME + "/" + TRAIN_SCRIPT,
            "sagemaker_container_log_level": "20",
            "sagemaker_job_name": MODEL_NAME + "-{{ ts_nodash }}",
            "sagemaker_region": REGION,
        },
        "TrainingJobName": MODEL_NAME + "-{{ ts_nodash }}",
    }
    return config


def generate_sagemaker_model_config(
    MODEL_NAME, MODEL_SCRIPT, CODE_DIR, REGION, MODEL_PATH, SM_ROLE, TRAIN_TASK_ID
):

    sub_dir = (
        "{{ ti.xcom_pull(task_ids='"
        + TRAIN_TASK_ID
        + "', key='return_value')['Training']['HyperParameters']['sagemaker_submit_directory'] }}"
    )
    print(sub_dir)
    # train_return_dict = yaml.load(train_return)
    config = {
        "ModelName": MODEL_NAME,
        "PrimaryContainer": {
            "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "Environment": {
                "SAGEMAKER_PROGRAM": MODEL_SCRIPT,
                "SAGEMAKER_SUBMIT_DIRECTORY": sub_dir,
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": REGION,
            },
            "ModelDataUrl": MODEL_PATH,
        },
        "ExecutionRoleArn": "" + SM_ROLE,
    }

    return config


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

    # MODEL_NAME = "credit-model"
    # BUCKET_NAME = "klarna-case-model-bucket"
    # SM_ROLE = "klarna-case-sm-role"
    # TRAIN_SCRIPT = "test.py"
    # REGION = "us-east-1"
    # QUERY_DATA_TASK_ID = "run_query_test"

    train_op = SageMakerTrainingOperator(
        task_id="test_training",
        config=generate_sagemaker_train_config(
            "credit-model",
            "klarna-case-model-bucket",
            "klarna-case-sm-role",
            "train.py",
            "us-east-1",
            "run_query_test",
        ),
        wait_for_completion=True,
        dag=dag,
    )

    create_model = SageMakerModelOperator(
        task_id="create_new_model",
        config=generate_sagemaker_model_config(
            "credit-model",
            "train.py",
            "",  ###################### FIX
            "us-east-1",
            "",  ############################FIX
            "klarna-case-sm-role",
            "test_training",
        ),
        dag=dag,
    )


test_athena_operator >> train_op >> create_model

if __name__ == "__main__":
    dag.cli()
