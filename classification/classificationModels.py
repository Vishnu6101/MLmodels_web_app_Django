import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

def LogisticReg():
    path = 'classification/mushroomsupdated.csv'
    data = pd.read_csv(path)

    dDict = defaultdict(LabelEncoder)
    cat_cols = data.dtypes == object
    cat_labels = data.columns[cat_cols].tolist()

    data[cat_labels] = data[cat_labels].apply(lambda col: dDict[col.name].fit_transform(col))

    Y = data.pop('class')
    X = data

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

    logreg = LogisticRegression(max_iter=1000)
    logModel = logreg.fit(X_train, Y_train)
    logPredict = logModel.predict(X_test)

    accuracy = round(accuracy_score(Y_test, logPredict), 4) * 100
    precision = round(precision_score(Y_test, logPredict), 4) * 100
    recall = round(recall_score(Y_test, logPredict), 4) * 100
    F_Score = round(f1_score(Y_test, logPredict), 4) * 100
    cm = confusion_matrix(Y_test, logPredict)

    return dict({'File' : path.split('/')[1], 'Accuracy' : accuracy, 'Precision' : precision, 'Recall' : recall, 'F_Score' : F_Score, 'Confusion Matrix' : cm, 'Data Columns' : data.columns.tolist()})