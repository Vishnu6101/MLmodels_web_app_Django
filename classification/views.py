from django.shortcuts import render, redirect
from django.urls import reverse
from . import classificationModels
from .forms import FileUploadForm
from .models import FileUpload

# Create your views here.
def LogisticReg(request):
    classificationResult = classificationModels.LogisticReg()
    # titles = list(['Accuracy', 'Precision', 'Recall', 'F_Score', 'Confusion Matrix', 'Data Columns', 'path'])
    context = {'result' : classificationResult}
    return render(request, 'logisticRegression.html', context)

def upload_file(request):
    if request.method == 'POST':
        fileform = FileUploadForm(request.POST, request.FILES)
        if fileform.is_valid():
            fileform.save()
            return redirect('Logistic_Reg')
    else:
        fileform = FileUploadForm()
    return render(request, 'file_upload.html', {'fileform': fileform})