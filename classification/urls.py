from django.urls import path
from . import views

urlpatterns = [
    path('logistic', views.LogisticRegressionModel, name='Logistic_Regression'),
    path('decisiontree', views.DecisionTreeClassifierModel, name="DecisionTree_Classifier"),
    path('randomforest', views.RandomForestClassifierModel, name="RandomForest_Classifier"),
    path('knn', views.KNNClassifierModel, name="KNN_Classifier"),
    path('upload/', views.upload_file, name='upload_file'),
]