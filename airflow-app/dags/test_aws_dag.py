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
from airflow.contrib.operators.ssh_operator import SSHOperator
from aws_operators import (
    DeployModelOperator,
    generate_sagemaker_train_config,
    generate_sagemaker_model_config,
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

    train_op = SageMakerTrainingOperator(
        task_id="test_training",
        config=generate_sagemaker_train_config(
            "credit_model",
            "klarna-case-model-bucket",
            "klarna-case-sm-role",
            "train.py",
            "us-east-1",
            "run_query_test",
        ),
        wait_for_completion=True,
        dag=dag,
    )

    deploy_model = DeployModelOperator(
        task_id="deploy_model",
        model_bucket="klarna-case-model-bucket",
        model_location="credit-model/model/model.joblib",
        new_version_location="credit-model/train/output-data/20220208/{{ ti.xcom_pull(task_ids='test_training', key='return_value')['Training']['TrainingJobName'] }}/output/model.tar.gz",
        model_filename="model.joblib",
    )

    redeploy_webserver = SSHOperator(
        task_id="redeploy_webserver",
        ssh_conn_id="master-node-ssh",
        command="kubectl replace --force -f kserve-configs/klarna-case-credit-model.yaml",
    )


test_athena_operator >> train_op >> deploy_model >> redeploy_webserver

if __name__ == "__main__":
    dag.cli()
