from django.shortcuts import render, redirect
from django.http import HttpResponse

from .regressionModels import Build_Regressor


# Create your views here.
def LinearRegressionModel(request):
    try:
        model = 'Linear Regression'
        dataset_url = request.session.get('dataset_url', None)

        if dataset_url:
            predictionResult = Build_Regressor(model=model, path=dataset_url)
        else:
            predictionResult = Build_Regressor(model=model)

        context = {'result' : predictionResult, 'Algorithm' : model}
        return render(request, 'regression.html', context)
    except Exception as e:
        return HTTPResponse(e)

def DecisionTreeRegressionModel(request):
    model = 'Decision Tree Regression'

    dataset_url = request.session.get('dataset_url', None)

    if dataset_url:
        predictionResult = Build_Regressor(model=model, path=dataset_url)
    else:
        predictionResult = Build_Regressor(model=model)

    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)

def RandomForestRegressionModel(request):
    model = 'Random Forest Regression'

    dataset_url = request.session.get('dataset_url', None)

    if dataset_url:
        predictionResult = Build_Regressor(model=model, path=dataset_url)
    else:
        predictionResult = Build_Regressor(model=model)
    
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)

def KNNRegressionModel(request):
    model = 'KNN Regression'

    dataset_url = request.session.get('dataset_url', None)

    if dataset_url:
        predictionResult = Build_Regressor(model=model, path=dataset_url)
    else:
        predictionResult = Build_Regressor(model=model)
    
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)