from django.shortcuts import render, redirect
from django.urls import reverse
from django.conf import settings
from django.http import HttpResponse

from .classificationModels import Build_Classifier

from .forms import FileUploadForm
from .models import FileUpload

import pandas as pd
import boto3
import os
import io

s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
                    # region_name=settings.AWS_S3_REGION_NAME
                )

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
            upload_location = os.path.join('files', file.name)
            s3_client.upload_fileobj(file, settings.AWS_STORAGE_BUCKET_NAME, upload_location)
            
            return redirect('Logistic_Regression')
    else:
        fileform = FileUploadForm()
    return render(request, 'file_upload.html', {'fileform': fileform})

def list_files(request):
    
    if request.method == 'POST':
        file_to_load = request.POST.get('selected_file')
        
        if file_to_load:
            s3_url = f's3://{settings.AWS_STORAGE_BUCKET_NAME}/{file_to_load}'
            
            try:
                # Read the CSV file from the S3 URL using pandas
                df = pd.read_csv(s3_url)
            except Exception as e:
                print("Error", e)

        return render(request, 'classification.html')
    
    else:
        s3_resource = boto3.resource(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        
        bucket = s3_resource.Bucket(settings.AWS_STORAGE_BUCKET_NAME)
        files = [obj.key for obj in bucket.objects.all()]
        print(files)
        
        return render(request, 'file_upload.html', {'files': files})

# TODO : 
# 2. list all the saved files
# 3. select the required file and load it from S3 and use it for model building