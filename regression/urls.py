from django.urls import path
from . import views

urlpatterns = [
    path('linear', views.LinearRegressionModel, name='Linear_Regression'),
    path('decisiontree', views.DecisionTreeRegressionModel, name="DecisionTree_Regression"),
    path('randomforest', views.RandomForestRegressionModel, name="RandomForest_Regression"),
    path('knn', views.KNNRegressionModel, name="KNN_Regression")
]