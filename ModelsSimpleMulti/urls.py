from django.urls import path
from . import views

app_name = 'ModelsSimpleMulti'


urlpatterns = [
    path('multi/', views.score_segment, name='modelsimplemulti'),
    path('chosse-model/', views.dataset, name='dataset'),
    path('get-resultat/', views.test_dataset),
    path('js-to-py/', views.jstopy),
    path('temperature-actuelle/', views.temperatureactuel),
    path('temperature-autre-ville/', views.changemodelresultat),
]