from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from datetime import datetime
from tempfile import NamedTemporaryFile


class DeployModelOperator(BaseOperator):
    template_fields = "model_location"

    template_ext = ()

    ui_color = "#e4e6f0"

    @apply_defaults
    def __init__(self, model_location=None, *args, **kwargs):
        self.model_location = model_location

    def execute(self, context):

        self.log.info(self.model_location)

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
