from django.db import models

# Create your models here.
# class CreateExperimentModel(models.Model):
#     name = models.CharField(max_length=20)

#     def __str__(self):
#         return self.name

class FileUpload(models.Model):
    upload = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)