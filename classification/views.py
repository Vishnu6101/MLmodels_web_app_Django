from django.shortcuts import render, redirect
from django.urls import reverse

from .classificationModels import Build_Classifier

from .forms import FileUploadForm
from .models import FileUpload

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

def upload_file(request):
    if request.method == 'POST':
        fileform = FileUploadForm(request.POST, request.FILES)
        if fileform.is_valid():
            fileform.save()
            return redirect('Logistic_Regression')
    else:
        fileform = FileUploadForm()
    return render(request, 'file_upload.html', {'fileform': fileform})


# TODO : 
# 1. save the file in S3
# 2. list all the saved files
# 3. select the required file and load it from S3 and use it for model building