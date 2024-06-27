from django import forms
from .models import FileUpload

# class CreateExperimentForm(forms.ModelForm):
#     class Meta:
#         model = CreateExperimentModel
#         fields = ('name',)

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = FileUpload
        fields = ('upload',)