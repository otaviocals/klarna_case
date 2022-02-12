import pandas as pd
import numpy as np
import math
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor

# Pre-processing Operator
class PreProc(BaseEstimator, TransformerMixin):
    def __init__(self, target_column="", fitted=False, encoders={}, geo_encoders={}):
        self.target_column = target_column
        self.fitted = fitted
        self.encoders = encoders
        self.geo_encoders = geo_encoders

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):

        # Set Target Feature
        target_column = self.target_column

        # Set Categorical & Geographic Features
        categoricals = ["merchant_category", "merchant_group"]
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
                        "order_lat",
                        "order_lng",
                        "promised_time",
                        "hour",
                        "on_demand",
                        "relat_vol",
                        "sum_un",
                        "sum_kg",
                        "seniority",
                        "found_rate",
                        "picking_speed",
                        "accepted_rate",
                        "rating",
                        "store_id",
                        "store_lat",
                        "store_lng",
                        "distance",
                    ]
                # Validation prediction (with target)
                elif len(X.columns) == 18:
                    print("Validation prediction")
                    X.columns = [
                        "order_lat",
                        "order_lng",
                        "promised_time",
                        "hour",
                        "on_demand",
                        "relat_vol",
                        "sum_un",
                        "sum_kg",
                        "seniority",
                        "found_rate",
                        "picking_speed",
                        "accepted_rate",
                        "rating",
                        "store_id",
                        "store_lat",
                        "store_lng",
                        "distance",
                        "total_minute",
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

        print("Training model")

        X_train, X_test, y_train, y_test = X

        # Set param_grid for hyperparameter tunning

        # SVR PARAMS
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

        # RFR PARAMS
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

        # LinearRegression
        # param_grid = {
        #        "fit_intercept": [
        #            True, # Best
        #            False
        #    ]
        # }
        # R2: 0.36929
        # RMSE: 28.031138

        # Selected Model
        # model_lib = SVR()
        # model_lib = RandomForestRegressor()
        # model_lib = XGBRegressor()
        model_lib = CatBoostRegressor()
        # model_lib = GaussianProcessRegressor()
        # model_lib = LinearRegression()

        # Tune hyperparameters and refit for best metrics
        grid_regressor = GridSearchCV(
            model_lib,
            param_grid=param_grid,
            cv=3,
            scoring=["r2", "neg_mean_squared_error"],
            refit="r2",
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
        y_test = y_test.reset_index(drop=True)

        # Validation metrics
        metrics = pd.DataFrame(
            {
                "RMSE": [math.sqrt(mean_squared_error(y_test, test_predictions))],
                "R2": [r2_score(y_test, test_predictions)],
            }
        )

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
