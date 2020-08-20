from django.urls import path
from . import views

app_name = 'ModelsSimpleMulti'


urlpatterns = [
    path('multi/', views.score_segment, name='modelsimplemulti'),
]