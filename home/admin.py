from django.contrib import admin
from .models import CreateExperimentModel, FileUpload

# Register your models here.
admin.site.register(CreateExperimentModel)
admin.site.register(FileUpload)
