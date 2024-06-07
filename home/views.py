from django.shortcuts import render
from .forms import CreateExperimentForm, FileUploadForm
from .models import CreateExperimentModel
from mlflow_utils import create_mlflow_experiment
import mlflow

import constant

# Create your views here.
# def home(request):
#     # new_experiment = create_mlflow_experiment(experiment_name="Experiment 1")
#     return render(request, 'home.html')


def create_exp(request):
    
    if request.method == 'POST':

        create_exp_form = CreateExperimentForm(request.POST)
        if create_exp_form.is_valid():
            create_exp_form.save()

            # create new exp
            text = create_exp_form.cleaned_data['name']
            # print(text)
            new_experiment = create_mlflow_experiment(experiment_name=text)
            constant.experiment_name = text
            return render(request, 'home.html', {"experiment" : constant.experiment_name})
    else:
        create_exp_form = CreateExperimentForm()
    return render(request, 'home.html', {'create_exp_form': create_exp_form, "experiment" : constant.experiment_name})

def upload_dataset(request):
    if request.method == 'POST':
        fileform = FileUploadForm(request.POST, request.FILES)
        if fileform.is_valid():
            fileform.save()

            # save it to the corresponding folder in s3
            return redirect('create_exp')
    else:
        fileform = FileUploadForm()
    return render(request, 'upload.html', {'fileform': fileform})