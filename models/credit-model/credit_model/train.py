import argparse
import datetime
import os
import subprocess
import sys
import json
import pandas as pd
import joblib
import re
import boto3
from sklearn import svm
import logging
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Install latest package version
MODULE_PATH = os.environ.get("SM_MODULE_DIR").strip("\n")
module_match = re.match("s3://(.*?)/(.*)", MODULE_PATH)
module_bucket = module_match.group(1)
module_location = module_match.group(2)


def download_data_webserver(bucket, source_filename, filename):
    s3 = boto3.client(
        "s3",
    )
    s3.download_file(bucket, source_filename, filename)
    logging.info("Downloaded data.")
    return


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


download_data_webserver(
    module_bucket,
    module_location,
    "credit_model-1.0.tar.gz",
)
install("credit_model-1.0.tar.gz")

from credit_model.libs import PreProc, Split, FeatSelect, Model


def train():

    # Load data
    input_files = os.listdir("../input/data/training")

    p = re.compile(".*\.csv$")
    input_file = [s for s in input_files if p.match(s)][0]

    # Create model operators pipeline
    pipe = Pipeline(
        [
            ("preproc", PreProc()),
            ("split", Split()),
            ("featselect", FeatSelect()),
            ("model", Model()),
        ]
    )

    # Set pipeline initial parameters
    params = {
        "preproc__fitted": False,
        "preproc__target_column": "default",
        "split__fitted": False,
        "split__target_column": "default",
        "split__split_size": 0.7,
        "featselect__fitted": False,
        "featselect__drop_features": 5,
        "model__fitted": False,
    }

    pipe.set_params(**params)

    # Get train data
    data = pd.read_csv("../input/data/training/" + input_file)

    # Train model
    with parallel_backend("threading"):
        pipe.fit(data)

    # Dump trained model
    joblib.dump(pipe, "../model/model.joblib", compress=2)

    # Dump model metrics
    metrics = pipe.get_params()["model__metrics"]
    metrics.to_csv("../model/metrics.csv", index=False)

    return


if __name__ == "__main__":
    train()
