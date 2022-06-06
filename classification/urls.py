from django.urls import path
from . import views

urlpatterns = [
    path('', views.LogisticReg, name='Logistic_Reg'),
]