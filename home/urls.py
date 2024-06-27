from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home_page'),
    # path('', views.create_exp, name='create_exp'),
    path('datasets/upload', views.upload_dataset, name='upload_dataset'),
    path('datasets/list', views.list_datasets, name='list_dataset')
]