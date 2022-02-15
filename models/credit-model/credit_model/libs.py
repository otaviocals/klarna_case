import os
import pandas as pd
import numpy as np
import math
import awswrangler as wr
from boto3.session import Session
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import GridSearchCV

# Pre-processing Operator
class PreProc(BaseEstimator, TransformerMixin):
    def __init__(
        self, target_column="", columns=None, fitted=False, encoders={}, geo_encoders={}
    ):
        self.target_column = target_column
        self.columns = columns
        self.fitted = fitted
        self.encoders = encoders
        self.geo_encoders = geo_encoders

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):

        # Set Target Feature
        target_column = self.target_column

        # Set Categorical & Geographic Features
        categoricals = [
            "merchant_category",
            "merchant_group",
            "name_in_email",
            "has_paid",
        ]
        geos = []

        print("Preprocessing data")

        # Train Pre-proc
        if not self.fitted:

            # Set Data Columns
            columns = list(X.columns)

            # Drop unused Features
            X = X.drop(["uuid"], axis=1)

            encoders = {}
            geo_encoders = {}

            # Encode Categoricals
            for categorical in categoricals:
                encoder = LabelEncoder()

                X[categorical] = encoder.fit_transform(X[categorical])

                encoders[categorical] = encoder

            # Cluster Geographicals (KMeans)
            for geo in geos:
                geo_data = X[[geo + "_lat", geo + "_lng"]]
                max_k = 10

                # Get best cluster number (elbow method)
                distortions = []
                for i in range(1, max_k + 1):
                    if len(geo_data) >= i:
                        model = KMeans(
                            n_clusters=i,
                            init="k-means++",
                            max_iter=300,
                            n_init=10,
                            random_state=0,
                        )
                        model.fit(geo_data)
                        distortions.append(model.inertia_)
                k = [i * 100 for i in np.diff(distortions, 2)].index(
                    min([i * 100 for i in np.diff(distortions, 2)])
                )

                # Refit with best cluster number
                final_geo = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    max_iter=300,
                    n_init=10,
                    random_state=0,
                ).fit(geo_data)

                # Create clustered feature
                X[geo + "_cluster"] = final_geo.predict(geo_data)
                # Drop unclustered features
                X = X.drop([geo + "_lat", geo + "_lng"], axis=1)

                # Save fitted clusterer
                geo_encoders[geo] = final_geo

            # Fill missing data
            X = X.fillna(0)

            # Set params for predict
            params = {
                "columns": columns,
                "encoders": encoders,
                "geo_encoders": geo_encoders,
                "fitted": True,
            }

            self.set_params(**params)

        else:

            # Setting up predict data

            if type(X).__name__ == "list":

                # Online Prediction
                if len(X) >= 1:
                    aws_session = Session(
                        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID").strip(
                            "\n"
                        ),
                        aws_secret_access_key=os.environ.get(
                            "AWS_SECRET_ACCESS_KEY"
                        ).strip("\n"),
                    )

                    ids = pd.DataFrame(X, columns=["uuid"])

                    query = "select * from credit_train_data where uuid IN {}"
                    X = wr.athena.read_sql_query(
                        query.format(
                            str(['"' + item + '"' for item in X])
                            .replace("[", "(")
                            .replace("]", ")")
                        ),
                        database="klarna_case",
                        boto3_session=aws_session,
                    )
                    X["uuid"] = X["uuid"].apply(lambda x: x.strip('"'))
                    X = X[self.columns]

                    # Handle missing IDs
                    unique_ids = set(ids["uuid"].unique())
                    print(unique_ids)
                    print(X["uuid"])

                    X["missing"] = X["uuid"].apply(
                        lambda x: 0 if x in unique_ids else 1
                    )
                    X = ids.merge(X, how="left", on="uuid")
                    X["missing"] = X["missing"].fillna(1)
                else:
                    raise

            if target_column in X.columns:
                X = X.drop([target_column], axis=1)

            # Drop unused Features
            X = X.drop(["uuid"], axis=1)

            # Load Categorical and Geographical Encoders
            encoders = self.encoders
            geo_encoders = self.geo_encoders

            # Encode Categorical Predict Features
            for categorical in categoricals:
                if categorical == target_column:
                    continue
                encoder = encoders[categorical]
                encoder_dict = dict(
                    zip(encoder.classes_, encoder.transform(encoder.classes_))
                )
                X[categorical] = X[categorical].apply(lambda x: encoder_dict.get(x, -1))

            # Encode Geographical Predict Features
            for geo in geos:
                X[geo + "_cluster"] = geo_encoders[geo].predict(
                    X[[geo + "_lat", geo + "_lng"]]
                )
                X = X.drop([geo + "_lat", geo + "_lng"], axis=1)

            # Fill missing data
            X = X.fillna(0)

            # Convert to numeric
            X = X.apply(pd.to_numeric)

        print("Preprocessing finished")

        return X


# Train-test split Operator
class Split(BaseEstimator, TransformerMixin):
    def __init__(self, target_column="", split_size=0.7, fitted=False):
        self.target_column = target_column
        self.split_size = split_size
        self.fitted = fitted

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):

        if not self.fitted:

            print("Spliting data")

            # Split target and features
            y = X[self.target_column]
            X = X.drop([self.target_column], axis=1)

            # Split train/test target and features
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=self.split_size
            )

            # Set params for predict
            params = {"fitted": True}

            self.set_params(**params)

            print("Spliting finished")

            return (X_train, X_test, y_train, y_test)

        else:
            return X


# Feature Selection Operator
class FeatSelect(BaseEstimator, TransformerMixin):
    def __init__(self, fitted=False, rank=None, select_features=None, drop_features=0):
        self.fitted = fitted
        self.rank = rank
        self.select_features = select_features
        self.drop_features = drop_features

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):

        # Train - Select best features
        if not self.fitted:

            print("Selecting features")

            X_train, X_test, y_train, y_test = X

            # Load Lasso regressor for feature selection
            selector = Lasso(alpha=0.001)
            # selector = Ridge(alpha=0.001)

            # Fit Lasso
            selector.fit(X_train, y_train)

            # Rank features
            rank = pd.Series(selector.coef_, index=X_train.columns).sort_values(
                ascending=False
            )

            print("Unfiltered Features")
            print(rank)

            # Select Features (drop 'drop_features' less relevant features)
            select_features = rank.abs().sort_values(ascending=False).index
            select_features = list(
                select_features[0 : len(select_features) - self.drop_features]
            )

            print("Selected Features")
            print(select_features)
            print("Dropped Features")
            print(list(set(list(X_train.columns)) - set(list(select_features))))

            # Filter selected features
            X_train = X_train[select_features]
            X_test = X_test[select_features]

            # Set params for predict
            params = {"rank": rank, "select_features": select_features, "fitted": True}

            self.set_params(**params)

            print("Selecting finished")

            return (X_train, X_test, y_train, y_test)
        # Predict - Filter best features
        else:

            # Load selected features
            features = self.select_features.copy()
            features.append("missing")
            print(features)

            # Filter selected features
            X = X[features]

            print("Filtering features")

            return X


# Model Operator
class Model(BaseEstimator, RegressorMixin):
    def __init__(self, fitted=False, model=None, metrics=None, cutoff=0.5):
        self.fitted = fitted
        self.cutoff = cutoff
        self.model = model
        self.metrics = metrics

    def fit(self, X=None, y=None):
        def cutoff_analysis(y_test, test_predictions_proba):

            metrics = pd.DataFrame(
                columns=[
                    "cutoff",
                    "vol_1",
                    "vol_0",
                    "roc_auc",
                    "recall",
                    "precision",
                    "f1",
                    "balanced_accuracy",
                ]
            )

            for i in range(0, 100):

                predictions_cutoff = i / 100

                test_predictions_proba_cut = test_predictions_proba.copy()
                test_predictions_proba_cut.loc[
                    test_predictions_proba_cut >= predictions_cutoff
                ] = 1
                test_predictions_proba_cut.loc[
                    test_predictions_proba_cut < predictions_cutoff
                ] = 0

                metric = pd.DataFrame(
                    {
                        "cutoff": [predictions_cutoff],
                        "vol_1": [
                            len(
                                test_predictions_proba_cut.loc[
                                    test_predictions_proba_cut == 1
                                ]
                            )
                        ],
                        "vol_0": [
                            len(
                                test_predictions_proba_cut.loc[
                                    test_predictions_proba_cut == 0
                                ]
                            )
                        ],
                        "roc_auc": [roc_auc_score(y_test, test_predictions_proba_cut)],
                        "recall": [recall_score(y_test, test_predictions_proba_cut)],
                        "precision": [
                            precision_score(y_test, test_predictions_proba_cut)
                        ],
                        "f1": [f1_score(y_test, test_predictions_proba_cut)],
                        "balanced_accuracy": [
                            balanced_accuracy_score(y_test, test_predictions_proba_cut)
                        ],
                    }
                )

                metrics = metrics.append(metric)

            return metrics

        print("Training model")

        X_train, X_test, y_train, y_test = X

        # Set param_grid for hyperparameter tunning

        # SVC PARAMS
        # param_grid = {
        #    "kernel": [
        #        "rbf",
        #        "linear",
        #        # "poly",
        #        "sigmoid",
        #    ],
        #    "C": [
        #        # 0.0,
        #        0.5,
        #        1.0,
        #        2.0,
        #    ],
        #    "gamma": ["scale", "auto"],
        # }
        # ROC_AUC:   0.
        # RECAL:     0.
        # F1:        0.
        # PRECISION: 0.
        # BAL_ACC:   0.

        # RFC PARAMS
        # param_grid = {
        #    "n_estimators": [
        #        #100,
        #        #300,
        #        1000 # BEST
        #        ],
        #    "min_samples_split": [
        #        #2,
        #        #10,
        #        40 # BEST
        #        ],
        #    "max_features": [
        #        "auto", # BEST
        #        #"sqrt"
        #        ],
        #    "bootstrap": [
        #        True, # BEST
        #        #False
        #        ],
        # }
        # ROC_AUC:   0.65
        # RECAL:     0.31
        # F1:        0.31
        # PRECISION: 0.30
        # BAL_ACC:   0.65

        # XGBoost Params
        param_grid = {
            "n_estimators": [
                500,  # BEST
                # 3000
            ],
            "learning_rate": [
                # 0.01,
                0.03,  # BEST
                # 0.07
            ],
            "max_depth": [
                4,  # BEST
                # 5,
                # 7
            ],
        }
        # ROC_AUC:   0.70
        # RECAL:     0.42
        # F1:        0.29
        # PRECISION: 0.23
        # BAL_ACC:   0.70

        # CatBoost Params
        # param_grid = {
        #    "depth": [
        #        4,  # BEST
        #        # 6,
        #        # 10
        #    ],
        #    "learning_rate": [
        #        # 0.01,
        #        0.1,  # BEST
        #        # 0.3
        #    ],
        #    "iterations": [
        #        # 30,
        #        # 100,
        #        400  # BEST
        #    ],
        #    "silent": [True],
        # }
        # ROC_AUC:   0.68
        # RECAL:     0.37
        # F1:        0.28
        # PRECISION: 0.22
        # BAL_ACC:   0.68

        # LogisticRegression
        # param_grid = {
        #    "fit_intercept": [
        #        True, # BEST
        #        #False
        #        ],
        #    "penalty": [
        #        "l1", # BEST
        #        #"l2",
        #        #"elasticnet"
        #        ],
        #    "class_weight": [
        #        "balanced", # BEST
        #        #None
        #        ],
        #    "solver": [
        #        #"newton-cg",
        #        #"lbfgs",
        #        "liblinear" # BEST
        #        ],
        # }
        # ROC_AUC:   0.65
        # RECAL:     0.33
        # F1:        0.24
        # PRECISION: 0.20
        # BAL_ACC:   0.65

        # Selected Model
        # model_lib = SVC()
        # model_lib = RandomForestClassifier()
        model_lib = XGBClassifier()
        # model_lib = CatBoostClassifier()
        # model_lib = LogisticRegression()

        # Tune hyperparameters and refit for best metrics
        grid_regressor = GridSearchCV(
            model_lib,
            param_grid=param_grid,
            cv=3,
            scoring=["roc_auc", "recall", "precision", "f1", "balanced_accuracy"],
            refit="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        grid_regressor.fit(X_train, y_train)

        # Log hyperparameter tuning metrics
        print("CV Results:")
        print(grid_regressor.cv_results_)
        print("Best Metrics:")
        print(grid_regressor.best_score_)
        print("Best Params:")
        print(grid_regressor.best_params_)

        # Get best model
        regressor = grid_regressor.best_estimator_

        # Validate model
        test_predictions = pd.Series(regressor.predict(X_test))
        test_predictions_proba = pd.Series(regressor.predict_proba(X_test)[:, 1])
        y_test = y_test.reset_index(drop=True)

        # Validation metrics
        metrics = cutoff_analysis(y_test, test_predictions_proba)
        metrics = metrics.sort_values(by="f1", ascending=False)
        best_cutoff = float(metrics.head(1)["cutoff"])
        print(metrics)
        print("Best cutoff: " + str(best_cutoff))

        metrics = cutoff_analysis(
            y_test,
            test_predictions_proba.apply(
                lambda x: 1 / (1 + np.exp(-10 * (x - best_cutoff)))
            ),
        )
        metrics = metrics.sort_values(by="f1", ascending=False)

        # Set params for predict
        params = {
            "model": regressor,
            "metrics": metrics,
            "fitted": True,
            "cutoff": best_cutoff,
        }

        self.set_params(**params)

        print("Training finished")

        return self

    def predict(self, X):

        print("Predicting data")

        # Load model
        regressor = self.model

        # Get missing
        X = X.reset_index()
        is_missing = X[["missing", "index"]]
        print(X)
        X = X.loc[X["missing"] == 0]
        X = X.drop(["missing", "index"], axis=1)

        # Convert to numeric
        X = X.astype(float)

        # Get predictions
        predictions = pd.DataFrame(
            regressor.predict_proba(X)[:, 1], columns=["predictions"]
        )
        predictions.index = X.index

        # Calibrate Predictions
        predictions = predictions.apply(
            lambda x: 1 / (1 + np.exp(-10 * (x - self.cutoff)))
        )

        # Merge Predictions to features
        print(is_missing)
        print(X)
        print(predictions)
        X = X.merge(predictions, left_index=True, right_index=True)
        X = is_missing.merge(X, how="left", left_index=True, right_index=True)
        X["predictions"] = X["predictions"].fillna(-1)
        print(X[["missing", "predictions"]])
        print(X["predictions"])

        print("Predicting finished")

        # Return predictions
        return X[["uuid", "predictions"]].to_numpy()
