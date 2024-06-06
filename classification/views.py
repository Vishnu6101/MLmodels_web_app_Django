from django.shortcuts import render, redirect
from django.urls import reverse
from django.conf import settings

from .classificationModels import Build_Classifier

from .forms import FileUploadForm
from .models import FileUpload

import boto3

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
            # fileform.save()
            # document = fileform.save(commit=False)
            file = request.FILES['upload']
            
            # Upload to S3
            s3 = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME
            )
            s3.upload_fileobj(file, settings.AWS_STORAGE_BUCKET_NAME, file.name)
            
            # Save the document to the database
            # document.save()
            return redirect('Logistic_Reg')
    else:
        fileform = FileUploadForm()
    return render(request, 'file_upload.html', {'fileform': fileform})


# TODO : 
# 1. save the file in S3
# 2. list all the saved files
# 3. select the required file and load it from S3 and use it for model building