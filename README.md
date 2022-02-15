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
    │   └── buildspec.yaml                  # CodePipeline configuration file of the CI/CD pipeline
    ├── airflow-app/                        # MLOps platform developed for the project
    │   ├── configs/                        # Kubernetes yaml configuration files of the MLOps environment
    │   ├── dags/                           # DAGs developed for model training and prediction
    │   └── plugins/                        # Custom Airflow Operators developed for interacting with AWS resources
    ├── models/                             # Models developed for the project
    │   └── credit-model/                   # Classification model developed for the case
    │       ├── credit_model/               # Model libs
    │       ├── kserve/                     # KServe endpoint cofiguration files
    │       └── train_image/                # Custom image for running custom training jobs in SageMaker
    ├── kserve-app/                         # KServe framework cofiguration files
    └── presentation/                       # Resources related to the project presentation
        └── eda_plots/                      # Distribution plots for EDA
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

