from django.shortcuts import render, redirect
from django.urls import reverse
from django.conf import settings

from .classificationModels import Build_Classifier

from .forms import FileUploadForm
from .models import FileUpload

from utils import S3_Connection

import os

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
            file = request.FILES['upload']
            
            # Upload to S3
            s3 = S3_Connection()
            upload_location = os.path.join('files', file.name)
            # print(upload_location)
            s3.upload_fileobj(file, settings.AWS_STORAGE_BUCKET_NAME, upload_location)
            
            return redirect('Logistic_Reg')
    else:
        fileform = FileUploadForm()
    return render(request, 'file_upload.html', {'fileform': fileform})


# TODO : 
# 2. list all the saved files
# 3. select the required file and load it from S3 and use it for model building