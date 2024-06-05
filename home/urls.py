from django.urls import path
from . import views

urlpatterns = [
    # path('', views.home, name='home_page'),
    path('', views.create_exp, name='create_exp')
]