from django.shortcuts import render, redirect

from .regressionModels import Build_Regressor

from classification.forms import FileUploadForm

# Create your views here.
def LinearRegressionModel(request):
    model = 'Linear Regression'
    predictionResult = Build_Regressor(model=model)
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)

def DecisionTreeRegressionModel(request):
    model = 'Decision Tree Regression'
    predictionResult = Build_Regressor(model=model)
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)

def RandomForestRegressionModel(request):
    model = 'Random Forest Regression'
    predictionResult = Build_Regressor(model=model)
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)

def KNNRegressionModel(request):
    model = 'KNN Regression'
    predictionResult = Build_Regressor(model=model)
    context = {'result' : predictionResult, 'Algorithm' : model}
    return render(request, 'regression.html', context)

def upload_file(request):
    if request.method == 'POST':
        fileform = FileUploadForm(request.POST, request.FILES)
        if fileform.is_valid():
            fileform.save()
            return redirect('Linear_Regression')
    else:
        fileform = FileUploadForm()
    return render(request, 'file_upload.html', {'fileform': fileform})