# Klarna Case

## About the Project

Challenge proposed by the Klarna team during 02/2022.
The development of a classification model was proposed for best estimating the credit default probability of custormers.

For this project the following were developed:
- A MLOps platform using Airflow, Kubernetes and AWS resources
- A classification model, which was deployed using KServe running on AWS EC2 and made available for either online or batch predictions

## Project Structure

* /cicd-pipeline - Resources built for the CI/CD pipeline of the project
* /kserve-app - KServe app for online model serving
* /airflow-app - Airflow platform built for managing the train and prediction workflows of the project
* /models - Developed models for the project
* /presentation - Resources related to the project presentation

### Detailed Structure

    .
    ├── cicd-pipeline/                      # Resources related to the CI/CD pipeline
    │   ├── Dockerfile                      # Dockerfile of the image used by the CI/CD pipeline
    │   ├── requirements.txt                # Requirements file for building the CI/CD container
    │   └── cloudbuild.yaml                 # Cloudbuild configuration file of the CI/CD pipeline
    ├── airflow_app/                        # MLOps platform developed for the project
    │   ├── configs/                        # Yaml configuration files of the MLOps environment
    │   │   ├── airflow_config.yaml         # Kubernetes resources configuration files
    │   │   └── helm_config.yaml            # Airflow Helm chart
    │   ├── dags/                           # DAGs developed for model training and prediction
    │   │   ├── klarna_model_train.py       # DAG for model training
    │   │   ├── klarna_model_predict.py     # DAG for generating model predictions
    │   │   └── sql/                        # SQL queries for data extraction used by the project's DAGs
    │   └── plugins/                        # Custom Airflow Operators developed for interacting with Google Cloud resources
    ├── models/                             # Models developed for the project
    │   └── credit-model/                   # Regression model developed for the case
    │       ├── setup.py                    # Script for packing the model
    │       └── credit_model/               # Developed libs for the model
    │           ├── gcp_utils.py            # Lib of functions for downloading and uploading data from GCS
    │           ├── train.py                # Script for building the model pipeline, fitting the model to the data and exporting results
    │           └── libs.py                 # Lib of operators for the steps of the model pipeline
    └── presentation/                       # Resources related to the project presentation
        ├── plot_metrics.py                 # Script for generating the graphics of the presentation
        ├── *.png                           # Image files of the presentation
        └── model_klarna.pdf                # Project presentation
## Architecture

Workflow architecture of the MLOps platform:
![alt text](presentation/arch_flow.png?raw=true)


Systems architecture of the MLOps environment:
![alt text](presentation/arch_sys.png?raw=true)

## Results

### Metrics
![alt text](presentation/metrics.png?raw=true)

### Feature Importances
![alt text](presentation/summary_plot1.png?raw=true)
![alt text](presentation/summary_plot2.png?raw=true)

