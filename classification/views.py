from django.shortcuts import render, redirect
from django.conf import settings

from .classificationModels import Build_Classifier

# Create your views here.
def LogisticRegressionModel(request):
    model = 'Logistic Regression'

    dataset_url = request.session.get('dataset_url', None)
    print(dataset_url)

    if dataset_url:
        classificationResult = Build_Classifier(model=model, path=dataset_url)
    else:
        classificationResult = Build_Classifier(model=model)
    
    context = {'result' : classificationResult, 'Algorithm' : model}
    return render(request, 'classification.html', context)

def DecisionTreeClassifierModel(request):
    model = "Decision Tree Classifier"

    dataset_url = request.session.get('dataset_url', None)
    print(dataset_url)

    if dataset_url:
        classificationResult = Build_Classifier(model=model, path=dataset_url)
    else:
        classificationResult = Build_Classifier(model=model)

    context = {'result' : classificationResult, 'Algorithm' : model}
    return render(request, 'classification.html', context)

def RandomForestClassifierModel(request):
    model = "Random Forest Classifier"

    dataset_url = request.session.get('dataset_url', None)
    print(dataset_url)

    if dataset_url:
        classificationResult = Build_Classifier(model=model, path=dataset_url)
    else:
        classificationResult = Build_Classifier(model=model)
    
    context = {'result' : classificationResult, 'Algorithm' : model}
    return render(request, 'classification.html', context)

def KNNClassifierModel(request):
    model = "KNN Classifier"
    
    dataset_url = request.session.get('dataset_url', None)
    print(dataset_url)

    if dataset_url:
        classificationResult = Build_Classifier(model=model, path=dataset_url)
    else:
        classificationResult = Build_Classifier(model=model)

    context = {'result' : classificationResult, 'Algorithm' : model}
    return render(request, 'classification.html', context)