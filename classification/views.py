from django.shortcuts import render
from . import classificationModels

# Create your views here.
def LogisticReg(request):
    classificationResult = classificationModels.LogisticReg()
    # titles = list(['Accuracy', 'Precision', 'Recall', 'F_Score', 'Confusion Matrix', 'Data Columns', 'path'])
    context = {'result' : classificationResult}
    return render(request, 'logisticRegression.html', context)