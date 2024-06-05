from django.shortcuts import render
from .forms import CreateExperimentForm
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
            return render(request, 'home.html')
    else:
        create_exp_form = CreateExperimentForm()
    return render(request, 'home.html', {'create_exp_form': create_exp_form})