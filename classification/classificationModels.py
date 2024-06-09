import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from mlflow_utils import get_mlflow_experiment
import mlflow

import constant

from utils import (
    extract_num_cat,
    calculate_metrics_classification,
    default_preprocessing_pipeline,
    classification_models
)

def Build_Classifier(model, path=None):
    experiment = get_mlflow_experiment(experiment_name=constant.experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # load the dataset
        if path is None:
            path = 'classification/mushroomsupdated.csv'
        
        data = pd.read_csv(path)

        X = data.drop(columns=['class'], axis = 1)
        Y = data['class']

        numerical_features, categorical_features = extract_num_cat(X)

        # Build the preprocessing pipeline
        preprocessing_pipeline = default_preprocessing_pipeline(numerical_features, categorical_features)

        preprocessing_pipeline.fit(X, Y)
        X_transformed = preprocessing_pipeline.transform(X)

        # label encode target variable
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)

        # train - test split
        X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, train_size=0.8, random_state=42)

        # fit the model
        classifier_model = classification_models[model]
        # print(classifier_model)
        classifier_model.fit(X_train, Y_train)
        logPredict = classifier_model.predict(X_test)

        # log model
        mlflow.sklearn.log_model(classifier_model, model)

        # # log params
        mlflow.log_params(classifier_model.get_params())

        # calculate the evaluation metrics
        accuracy, precision, recall, f_score, cm = calculate_metrics_classification(Y_test, logPredict)

        # log metrics
        mlflow.log_metrics(
            {"accuracy" : accuracy,
            "precision" : precision,
            "recall" : recall, 
            "f1_score" : f_score}
        )

        return dict({
            'File' : path.split('/')[-1], 
            'Accuracy' : accuracy, 
            'Precision' : precision, 
            'Recall' : recall, 
            'F_Score' : f_score, 
            'Confusion Matrix' : cm, 
            'Data Columns' : data.columns.tolist()
        })