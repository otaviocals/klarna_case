import kserve
import boto3
from typing import Dict
import joblib
import pandas as pd
import subprocess
import sys
import os
import warnings
import logging

warnings.simplefilter(action="ignore", category=FutureWarning)

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID").strip("\n")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY").strip("\n")


def download_data_webserver(bucket, source_filename, filename):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    s3.download_file(bucket, source_filename, filename)
    logging.info("Downloaded data.")
    return


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


class Model(kserve.KFModel):
    def __init__(self, name, model=""):
        super().__init__(name)
        self.name = name
        self.model = self.load()

    def load(self):
        download_data_webserver(
            "klarna-case-model-bucket",
            "credit-model/model/model.joblib",
            "model.joblib",
        )
        return joblib.load("model.joblib")

    def predict(self, request):
        columns = self.model.get_params()["preproc__columns"]
        data = pd.DataFrame(request["instances"], columns=columns)
        predictions = self.model.predict(data)
        return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    model = Model("credit-model")
    kserve.KFServer().start([model])
