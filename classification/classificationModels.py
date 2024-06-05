import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from mlflow_utils import get_mlflow_experiment
import mlflow

import constant

def calculate_metrics(Y_test, logPredict):
    accuracy = round(accuracy_score(Y_test, logPredict), 4) * 100
    precision = round(precision_score(Y_test, logPredict), 4) * 100
    recall = round(recall_score(Y_test, logPredict), 4) * 100
    f_score = round(f1_score(Y_test, logPredict), 4) * 100
    cm = confusion_matrix(Y_test, logPredict)
    
    return accuracy, precision, recall, f_score, cm

def Logistic_Regression():
    experiment = get_mlflow_experiment(experiment_name=constant.experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # load the dataset
        path = 'classification/mushroomsupdated.csv'
        data = pd.read_csv(path)

        X = data.drop(columns=['class'], axis = 1)
        Y = data['class']

        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Build the preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', OrdinalEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        preprocessing = Pipeline(steps=[
            ('preprocessor', preprocessor),
        ])

        preprocessing.fit(X, Y)
        X_transformed = preprocessing.transform(X)

        # label encode target variable
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)

        # train - test split
        X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, train_size=0.8, random_state=42)

        # fit the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)
        logPredict = model.predict(X_test)

        # log model
        mlflow.sklearn.log_model(model, "Logistic Regression")

        # # log params
        mlflow.log_params(model.get_params())

        # calculate the evaluation metrics
        accuracy, precision, recall, f_score, cm = calculate_metrics(Y_test, logPredict)

        # log metrics
        mlflow.log_metrics(
            {"accuracy" : accuracy,
            "precision" : precision,
            "recall" : recall, 
            "f1_score" : f_score}
        )

        return dict({'File' : path.split('/')[1], 'Accuracy' : accuracy, 'Precision' : precision, 'Recall' : recall, 'F_Score' : f_score, 'Confusion Matrix' : cm, 'Data Columns' : data.columns.tolist()})