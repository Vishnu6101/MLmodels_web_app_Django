from django import forms
from .models import CreateExperimentModel

class CreateExperimentForm(forms.ModelForm):
    class Meta:
        model = CreateExperimentModel
        fields = ('name',)