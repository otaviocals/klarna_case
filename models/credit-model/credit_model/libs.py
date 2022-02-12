import pandas as pd
import numpy as np
import math
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

        # Set Data Columns
        columns = list(X.columns)

        # Set Target Feature
        target_column = self.target_column

        # Set Categorical & Geographic Features
        categoricals = ["merchant_category", "merchant_group", "name_in_email"]
        geos = []

        print("Preprocessing data")

        # Train Pre-proc
        if not self.fitted:

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
                X = pd.DataFrame(X)

                # Batch prediction (no target)
                if len(X.columns) == 17:
                    print("Batch prediction")
                    X.columns = [
                        "uuid",
                        "account_amount_added_12_24m",
                        "account_days_in_dc_12_24m",
                        "account_days_in_rem_12_24m",
                        "account_days_in_term_12_24m",
                        "account_incoming_debt_vs_paid_0_24m",
                        "account_status",
                        "account_worst_status_0_3m",
                        "account_worst_status_12_24m",
                        "account_worst_status_3_6m",
                        "account_worst_status_6_12m",
                        "age",
                        "avg_payment_span_0_12m",
                        "avg_payment_span_0_3m",
                        "merchant_category",
                        "merchant_group",
                        "has_paid",
                        "max_paid_inv_0_12m",
                        "max_paid_inv_0_24m",
                        "name_in_email",
                        "num_active_div_by_paid_inv_0_12m",
                        "num_active_inv",
                        "num_arch_dc_0_12m",
                        "num_arch_dc_12_24m",
                        "num_arch_ok_0_12m",
                        "num_arch_ok_12_24m",
                        "num_arch_rem_0_12m",
                        "num_arch_written_off_0_12m",
                        "num_arch_written_off_12_24m",
                        "num_unpaid_bills",
                        "status_last_archived_0_24m",
                        "status_2nd_last_archived_0_24m",
                        "status_3rd_last_archived_0_24m",
                        "status_max_archived_0_6_months",
                        "status_max_archived_0_12_months",
                        "status_max_archived_0_24_months",
                        "recovery_debt",
                        "sum_capital_paid_account_0_12m",
                        "sum_capital_paid_account_12_24m",
                        "sum_paid_inv_0_12m",
                        "time_hours",
                        "worst_status_active_inv",
                    ]
                # Validation prediction (with target)
                elif len(X.columns) == 18:
                    print("Validation prediction")
                    X.columns = [
                        "uuid",
                        "default",
                        "account_amount_added_12_24m",
                        "account_days_in_dc_12_24m",
                        "account_days_in_rem_12_24m",
                        "account_days_in_term_12_24m",
                        "account_incoming_debt_vs_paid_0_24m",
                        "account_status",
                        "account_worst_status_0_3m",
                        "account_worst_status_12_24m",
                        "account_worst_status_3_6m",
                        "account_worst_status_6_12m",
                        "age",
                        "avg_payment_span_0_12m",
                        "avg_payment_span_0_3m",
                        "merchant_category",
                        "merchant_group",
                        "has_paid",
                        "max_paid_inv_0_12m",
                        "max_paid_inv_0_24m",
                        "name_in_email",
                        "num_active_div_by_paid_inv_0_12m",
                        "num_active_inv",
                        "num_arch_dc_0_12m",
                        "num_arch_dc_12_24m",
                        "num_arch_ok_0_12m",
                        "num_arch_ok_12_24m",
                        "num_arch_rem_0_12m",
                        "num_arch_written_off_0_12m",
                        "num_arch_written_off_12_24m",
                        "num_unpaid_bills",
                        "status_last_archived_0_24m",
                        "status_2nd_last_archived_0_24m",
                        "status_3rd_last_archived_0_24m",
                        "status_max_archived_0_6_months",
                        "status_max_archived_0_12_months",
                        "status_max_archived_0_24_months",
                        "recovery_debt",
                        "sum_capital_paid_account_0_12m",
                        "sum_capital_paid_account_12_24m",
                        "sum_paid_inv_0_12m",
                        "time_hours",
                        "worst_status_active_inv",
                    ]

            if target_column in X.columns:
                X = X.drop([target_column], axis=1)

            # Drop unused Features
            X = X.drop(["promised_time"], axis=1)

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
            features = self.select_features

            # Filter selected features
            X = X[features]

            print("Filtering features")

            return X


# Model Operator
class Model(BaseEstimator, RegressorMixin):
    def __init__(self, fitted=False, model=None, metrics=None):
        self.fitted = fitted
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
        #        "linear",    # BEST
        #        # "poly",
        #        "sigmoid",
        #    ],
        #    "C": [
        #        # 0.0,
        #        0.5,
        #        1.0,   # BEST
        #        2.0,
        #    ],
        #    "epsilon": [
        #        0.1,  # BEST
        #        0.2,
        #        0.5
        #    ],
        #    "gamma": [
        #        "scale", # BEST
        #        "auto"
        #    ],
        # }
        # R2: 0.358622
        # RMSE: 28.073866

        # RFC PARAMS
        # param_grid = {
        #    "n_estimators": [
        #        100,
        #        300,
        #        1000 # BEST
        #    ],
        #    "min_samples_split": [
        #        2, # BEST
        #        10,
        #        40
        #    ],
        #    "max_features": [
        #        "auto",
        #        "sqrt" # BEST
        #    ],
        #    "bootstrap": [
        #        True,
        #        False # BEST
        #    ],
        # }
        # R2:  0.493913
        # RMSE: 24.450065

        # XGBoost Params
        # param_grid = {
        #    "n_estimators": [
        #        500, # BEST
        #        3000
        #    ],
        #    "learning_rate": [
        #        0.01,
        #        0.03, # BEST
        #        0.07
        #    ],
        #    "max_depth": [
        #        4, # BEST
        #        5,
        #        7
        #    ],
        # }
        # R2: 0.511957
        # RMSE:  24.372637

        # CatBoost Params
        param_grid = {
            "depth": [4, 6, 10],  # BEST
            "learning_rate": [0.01, 0.1, 0.3],  # BEST
            "iterations": [30, 100, 400],  # BEST
            "silent": [True],
        }
        # R2: 0.555919
        # RMSE:   23.439524

        # GaussianMixture
        # param_grid = {
        #        "normalize_y": [
        #            True, # Best
        #            False
        #    ]
        # }
        # R2: 0.111585
        # RMSE: 31.622464

        # LogisticRegression
        # param_grid = {
        #        "fit_intercept": [
        #            True, # Best
        #            False
        #    ]
        # }
        # R2: 0.36929
        # RMSE: 28.031138

        # Selected Model
        # model_lib = SVC()
        # model_lib = RandomForestClassifier()
        # model_lib = XGBClassifier()
        model_lib = CatBoostClassifier()
        # model_lib = GaussianProcessClassifier()
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
        # metrics = pd.DataFrame(
        #    {
        #        "roc_auc": [roc_auc_score(y_test, test_predictions_proba)],
        #        "recall": [recall_score(y_test, test_predictions)],
        #        "precision": [precision_score(y_test, test_predictions)],
        #        "f1": [f1_score(y_test, test_predictions)],
        #        "balanced_accuracy": [
        #            balanced_accuracy_score(y_test, test_predictions)
        #        ],
        #    }
        # )

        metrics = cutoff_analysis(y_test, test_predictions_proba)

        print(metrics)

        # Set params for predict
        params = {"model": regressor, "metrics": metrics, "fitted": True}

        self.set_params(**params)

        print("Training finished")

        return self

    def predict(self, X):

        print("Predicting data")

        # Load model
        regressor = self.model

        # Get predictions
        predictions = pd.DataFrame(regressor.predict(X), columns=["predictions"])

        # Merge Predictions to features
        X = X.merge(predictions, left_index=True, right_index=True)

        print("Predicting finished")

        # Return predictions
        return X["predictions"].to_numpy()
