import argparse
import datetime
import os
import subprocess
import sys
import json
import pandas as pd
import joblib
import re
from sklearn import svm
import logging
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend
from libs import PreProc, Split, FeatSelect, Model
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def train():

    # Load data
    input_files = os.listdir("../input/data/training")
    logging.info(input_files)
    print(input_files)

    p = re.compile("\.csv$")
    input_file = [s for s in input_files if p.match(s)][0]

    logging.info(input_files)
    logging.info(input_file)

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
        "featselect__drop_features": 0,
        "model__fitted": False,
    }

    pipe.set_params(**params)

    # Get train data
    data = pd.read_csv(input_file)

    # Train model
    with parallel_backend("threading"):
        pipe.fit(data)

    # Dump trained model
    joblib.dump(pipe, "../model/model.joblib", compress=2)

    # Dump model metrics
    metrics = pipe.get_params()["model__metrics"]
    metrics["ref_date"] = arguments["date"]
    metrics.to_csv("metrics.csv", index=False)

    return


if __name__ == "__main__":
    train()
