from django.shortcuts import render, redirect
from django.urls import reverse

from .classificationModels import (
    Logistic_Regression, 
    DecisionTree_Classifier, 
    RandomForest_Classifier,
    KNN_Classifier)


from .forms import FileUploadForm
from .models import FileUpload

# Create your views here.
def LogisticRegressionModel(request):
    classificationResult = Logistic_Regression()
    context = {'result' : classificationResult, 'Algorithm' : 'Logistic Regression'}
    return render(request, 'classification.html', context)

def DecisionTreeClassifierModel(request):
    classificationResult = DecisionTree_Classifier()
    context = {'result' : classificationResult, 'Algorithm' : 'DecisionTree Classifier'}
    return render(request, 'classification.html', context)

def RandomForestClassifierModel(request):
    classificationResult = RandomForest_Classifier()
    context = {'result' : classificationResult, 'Algorithm' : 'RandomForest Classifier'}
    return render(request, 'classification.html', context)

def KNNClassifierModel(request):
    classificationResult = KNN_Classifier()
    context = {'result' : classificationResult, 'Algorithm' : 'K-Nearest Neighbors Classifier'}
    return render(request, 'classification.html', context)

def upload_file(request):
    if request.method == 'POST':
        fileform = FileUploadForm(request.POST, request.FILES)
        if fileform.is_valid():
            fileform.save()
            return redirect('Logistic_Reg')
    else:
        fileform = FileUploadForm()
    return render(request, 'file_upload.html', {'fileform': fileform})


# TODO : 
# 1. save the file in S3
# 2. list all the saved files
# 3. select the required file and load it from S3 and use it for model building