from django.shortcuts import render, redirect
from django.conf import settings

from .classificationModels import Build_Classifier

# Create your views here.
def LogisticRegressionModel(request):
    model = 'Logistic Regression'
    classificationResult = Build_Classifier(model=model)
    context = {'result' : classificationResult, 'Algorithm' : model}
    return render(request, 'classification.html', context)

def DecisionTreeClassifierModel(request):
    model = "Decision Tree Classifier"
    classificationResult = Build_Classifier(model=model)
    context = {'result' : classificationResult, 'Algorithm' : model}
    return render(request, 'classification.html', context)

def RandomForestClassifierModel(request):
    model = "Random Forest Classifier"
    classificationResult = Build_Classifier(model=model)
    context = {'result' : classificationResult, 'Algorithm' : model}
    return render(request, 'classification.html', context)

def KNNClassifierModel(request):
    model = "KNN Classifier"
    classificationResult = Build_Classifier(model=model)
    context = {'result' : classificationResult, 'Algorithm' : model}
    return render(request, 'classification.html', context)