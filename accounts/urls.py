from django.contrib import admin
from django.urls import path

from . import views

app_name = 'accounts'

urlpatterns = [
    path('profile/', views.profil, name='profil'),
    path('adapter-model/', views.adapter_model, name='adapter_model'),
    path('get-parametre-model/', views.get_parametre_model, name='get_parametre_model'),
    path('transfer-learning/', views.transfer_learning, name='transfer_learning'),
]