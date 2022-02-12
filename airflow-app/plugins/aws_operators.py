from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.hooks.base import BaseHook
from datetime import datetime
from tempfile import NamedTemporaryFile, TemporaryDirectory
import boto3
import tarfile


class DeployModelOperator(BaseOperator):
    template_fields = (
        "model_location",
        "new_version_location",
        "master_model_resource_path",
    )

    template_ext = ()

    ui_color = "#e4e6f0"

    @apply_defaults
    def __init__(
        self,
        model_bucket,
        model_location,
        new_version_location,
        model_filename="model.joblib",
        restart_model_webserver=False,
        master_internal_ip=None,
        master_model_resource_path=None,
        aws_conn_id="aws_default",
        aws_region="us-east-1",
        *args,
        **kwargs
    ):
        super(DeployModelOperator, self).__init__(*args, **kwargs)
        self.model_bucket = model_bucket
        self.model_location = model_location
        self.new_version_location = new_version_location
        self.model_filename = model_filename
        self.restart_model_webserver = restart_model_webserver
        self.master_internal_ip = master_internal_ip
        self.master_model_resource_path = master_model_resource_path
        self.aws_conn_id = aws_conn_id
        self.aws_region = aws_region

    def execute(self, context):

        aws_connection = BaseHook.get_connection(self.aws_conn_id)

        with NamedTemporaryFile() as tmp:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_connection.login,
                aws_secret_access_key=aws_connection.password,
                region_name=self.aws_region,
            )
            s3.download_file(self.model_bucket, self.new_version_location, tmp.name)

            with TemporaryDirectory() as tmp_dir:
                tarfile.open(tmp.name, mode="r:gz").extractall(tmp_dir)
                s3.upload_file(
                    tmp_dir + "/" + self.model_filename,
                    self.model_bucket,
                    self.model_location,
                )

        self.log.info("New version deployed")

        if self.restart_model_webserver:

            self.log.info("Restarting webserver")
            self.log.info("Webserver restarted")

        return


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


def generate_sagemaker_model_config(MODEL_NAME, MODEL_SCRIPT, TRAIN_TASK_ID):

    sub_dir = (
        "{{ ti.xcom_pull(task_ids='"
        + TRAIN_TASK_ID
        + "', key='return_value')['Training']['HyperParameters']['sagemaker_submit_directory'].strip('\\\"') }}"
    )

    region = (
        "{{ ti.xcom_pull(task_ids='"
        + TRAIN_TASK_ID
        + "', key='return_value')['Training']['HyperParameters']['sagemaker_region'].strip('\\\"') }}"
    )

    program = (
        "{{ ti.xcom_pull(task_ids='"
        + TRAIN_TASK_ID
        + "', key='return_value')['Training']['HyperParameters']['sagemaker_program'].strip('\\\"') }}"
    )

    model_path = (
        "{{ ti.xcom_pull(task_ids='"
        + TRAIN_TASK_ID
        + "', key='return_value')['Training']['ModelArtifacts']['S3ModelArtifacts'].strip('\\\"') }}"
    )

    sm_role = (
        "{{ ti.xcom_pull(task_ids='"
        + TRAIN_TASK_ID
        + "', key='return_value')['Training']['RoleArn'].strip('\\\"') }}"
    )

    config = {
        "ModelName": MODEL_NAME,
        "PrimaryContainer": {
            "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "Environment": {
                "SAGEMAKER_PROGRAM": MODEL_SCRIPT,
                "SAGEMAKER_SUBMIT_DIRECTORY": sub_dir,
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": region,
            },
            "ModelDataUrl": model_path,
        },
        "ExecutionRoleArn": sm_role,
    }

    return config
