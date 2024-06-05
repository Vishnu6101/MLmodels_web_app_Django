from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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