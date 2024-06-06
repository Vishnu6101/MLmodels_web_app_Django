import numpy as np

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    accuracy_score, 
    f1_score, 
    confusion_matrix, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

classification_models = {
    "Logistic Regression" : LogisticRegression(),
    "Decision Tree Classifier" : DecisionTreeClassifier(),
    "Random Forest Classifier" : RandomForestClassifier(),
    "KNN Classifier" : KNeighborsClassifier()
}

regression_models = {
    "Linear Regression" : LinearRegression(),
    "Decision Tree Regression" : DecisionTreeRegressor(),
    "Random Forest Regression" : RandomForestRegressor(),
    "KNN Regression" : KNeighborsRegressor()
}

def extract_num_cat(data):
    numerical_features = data.select_dtypes(include=['number']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()

    return numerical_features, categorical_features

def calculate_metrics_classification(Y_test, pred):
    accuracy = round(accuracy_score(Y_test, pred), 4) * 100
    precision = round(precision_score(Y_test, pred), 4) * 100
    recall = round(recall_score(Y_test, pred), 4) * 100
    f_score = round(f1_score(Y_test, pred), 4) * 100
    cm = confusion_matrix(Y_test, pred)
    
    return accuracy, precision, recall, f_score, cm

def calculate_metrics_regression(Y_test, pred):
    mae = mean_absolute_error(Y_test, pred)
    mse = mean_squared_error(Y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, pred)

    return mae, mse, rmse, r2

def default_preprocessing_pipeline(numerical_features, categorical_features):
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
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

    return preprocessing