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
    dag_id="klarna_refresh_docker_cred",
    default_args=args,
    schedule_interval="30 * * * *",
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=60),
) as dag:

    refresh_cred = SSHOperator(
        task_id="redeploy_webserver",
        ssh_conn_id="master-node-ssh",
        command="./auto_aws_auth.sh",
    )


refresh_cred

if __name__ == "__main__":
    dag.cli()
