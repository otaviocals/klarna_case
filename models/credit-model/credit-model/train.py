import argparse
import datetime
import os
import subprocess
import sys
import json
import pandas as pd
import joblib
from sklearn import svm
import logging
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def train():

    # Load data
    print(os.listdir(".."))
    print(os.listdir("../input/data/training"))

    # Create model operators pipeline
    pipe = Pipeline(steps=[])

    # Set pipeline initial parameters
    params = {}

    pipe.set_params(**params)

    # Get train data
    # data = pd.read_csv("train_data.csv")

    # Train model
    # with parallel_backend("threading"):
    #    pipe.fit(data)

    # Dump trained model
    joblib.dump(pipe, "model/model.joblib", compress=2)

    # Dump model metrics
    # metrics = pipe.get_params()["model__metrics"]
    # metrics["ref_date"] = arguments["date"]
    # metrics.to_csv("metrics.csv", index=False)

    return


if __name__ == "__main__":
    train()
