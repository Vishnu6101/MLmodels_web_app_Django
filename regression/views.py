from django.shortcuts import render, redirect

from .regressionModels import Build_Regressor


# Create your views here.
def LinearRegressionModel(request):
    model = 'Linear Regression'
    predictionResult = Build_Regressor(model=model)
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)

def DecisionTreeRegressionModel(request):
    model = 'Decision Tree Regression'
    predictionResult = Build_Regressor(model=model)
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)

def RandomForestRegressionModel(request):
    model = 'Random Forest Regression'
    predictionResult = Build_Regressor(model=model)
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)

def KNNRegressionModel(request):
    model = 'KNN Regression'
    predictionResult = Build_Regressor(model=model)
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)