import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from mlflow_utils import get_mlflow_experiment
import mlflow

import constant

from utils import (
    extract_num_cat,
    calculate_metrics_regression,
    default_preprocessing_pipeline,
    regression_models
)

def Build_Regressor(model, path=None):
    experiment = get_mlflow_experiment(experiment_name=constant.experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # load the dataset
        if path is None:
            path = 'regression/car details v4.csv'

        print(path)
        
        data = pd.read_csv(path)

        X = data.drop(columns=['Price'], axis = 1)
        Y = data['Price']

        numerical_features, categorical_features = extract_num_cat(X)
        # print(numerical_features, categorical_features)

        # Build the preprocessing pipeline
        preprocessing_pipeline = default_preprocessing_pipeline(numerical_features, categorical_features)
        # print(preprocessing_pipeline)

        preprocessing_pipeline.fit(X, Y)
        X_transformed = preprocessing_pipeline.transform(X)

        # train - test split
        X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, train_size=0.8, random_state=42)
        # print(X_train, X_test)

        # print(Y_train)

        # fit the model
        prediction_model = regression_models[model]
        # print(prediction_model)
        prediction_model.fit(X_train, Y_train)
        predictions = prediction_model.predict(X_test)

        # print(predictions)

        # log model
        mlflow.sklearn.log_model(prediction_model, model)

        # # log params
        mlflow.log_params(prediction_model.get_params())

        # calculate the evaluation metrics
        mae, mse, rmse, r2 = calculate_metrics_regression(Y_test, predictions)

        # print(mse)

        # log metrics
        mlflow.log_metrics(
            {"Mean Square Error" : mse,
            "Mean Absolute Error" : mae,
            "Root Mean Square Error" : rmse, 
            "R Squared" : r2}
        )

        return dict({
            'File' : path.split('/')[-1], 
            "Mean Square Error" : mse,
            "Mean Absolute Error" : mae,
            "Root Mean Square Error" : rmse, 
            "R Squared" : r2,
            'Data Columns' : data.columns.tolist()
        })
