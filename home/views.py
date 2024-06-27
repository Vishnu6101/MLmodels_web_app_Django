from django.shortcuts import render, redirect
from .forms import FileUploadForm
from django.conf import settings

import constant

import pandas as pd
import boto3
import os

s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )

def home(request):
    return render(request, 'home.html')

# def create_exp(request):
    
#     if request.method == 'POST':

#         create_exp_form = CreateExperimentForm(request.POST)
#         if create_exp_form.is_valid():
#             create_exp_form.save()

#             # create new exp
#             text = create_exp_form.cleaned_data['name']
#             # print(text)
#             constant.experiment_name = text
#             return render(request, 'home.html', {"experiment" : constant.experiment_name})
#     else:
#         create_exp_form = CreateExperimentForm()
#     return render(request, 'home.html', {'create_exp_form': create_exp_form, "experiment" : constant.experiment_name})

def upload_dataset(request):
    if request.method == 'POST':
        fileform = FileUploadForm(request.POST, request.FILES)
        if fileform.is_valid():

            # save it to the corresponding folder in s3
            file = request.FILES['upload']
            folder = request.POST.get('task')
            
            # Upload to S3
            upload_location = os.path.join(folder, file.name)
            s3_client.upload_fileobj(file, settings.AWS_STORAGE_BUCKET_NAME, upload_location)

            return redirect('home_page')
    else:
        fileform = FileUploadForm()
    return render(request, 'upload.html', {'fileform': fileform})

def list_datasets(request):
    if request.method == 'POST':
        file_to_load = request.POST.get('selected_file')
        
        if file_to_load:
            s3_url = f's3://{settings.AWS_STORAGE_BUCKET_NAME}/{file_to_load}'
            request.session['dataset_url'] = s3_url

        return redirect('home_page')
    
    else:
        s3_resource = boto3.resource(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        
        bucket = s3_resource.Bucket(settings.AWS_STORAGE_BUCKET_NAME)
        files = [obj.key for obj in bucket.objects.all()]
        
        return render(request, 'list_datasets.html', {'files': files})