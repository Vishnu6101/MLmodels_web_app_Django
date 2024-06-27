import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import constant

from utils import (
    extract_num_cat,
    calculate_metrics_regression,
    default_preprocessing_pipeline,
    regression_models
)

def Build_Regressor(model, path=None):
    # load the dataset
    if path is None:
        path = 'regression/car details v4.csv'

    print(path)
    
    data = pd.read_csv(path)

    X = data.drop(columns=['Price'], axis = 1)
    Y = data['Price']

    numerical_features, categorical_features = extract_num_cat(X)

    # Build the preprocessing pipeline
    preprocessing_pipeline = default_preprocessing_pipeline(numerical_features, categorical_features)

    preprocessing_pipeline.fit(X, Y)
    X_transformed = preprocessing_pipeline.transform(X)

    # train - test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, train_size=0.8, random_state=42)

    # fit the model
    prediction_model = regression_models[model]
    # print(prediction_model)
    prediction_model.fit(X_train, Y_train)
    predictions = prediction_model.predict(X_test)

    # calculate the evaluation metrics
    mae, mse, rmse, r2 = calculate_metrics_regression(Y_test, predictions)

    return dict({
        'File' : path.split('/')[-1], 
        "Mean Square Error" : mse,
        "Mean Absolute Error" : mae,
        "Root Mean Square Error" : rmse, 
        "R Squared" : r2,
        'Data Columns' : data.columns.tolist()
    })
