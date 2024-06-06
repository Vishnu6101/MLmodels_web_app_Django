from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import boto3
from django.conf import settings

classification_models = {
    "Logistic Regression" : LogisticRegression(),
    "Decision Tree Classifier" : DecisionTreeClassifier(),
    "Random Forest Classifier" : RandomForestClassifier(),
    "KNN Classifier" : KNeighborsClassifier()
}

def S3_Connection():
    s3 = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
                # region_name=settings.AWS_S3_REGION_NAME
            )
    
    return s3

def extract_num_cat(data):
    numerical_features = data.select_dtypes(include=['number']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()

    return numerical_features, categorical_features

def calculate_metrics(Y_test, pred):
    accuracy = round(accuracy_score(Y_test, pred), 4) * 100
    precision = round(precision_score(Y_test, pred), 4) * 100
    recall = round(recall_score(Y_test, pred), 4) * 100
    f_score = round(f1_score(Y_test, pred), 4) * 100
    cm = confusion_matrix(Y_test, pred)
    
    return accuracy, precision, recall, f_score, cm

def default_preprocessing_pipeline(numerical_features, categorical_features):
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

    return preprocessing