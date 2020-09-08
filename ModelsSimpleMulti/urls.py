from django.urls import path
from . import views

app_name = 'ModelsSimpleMulti'


urlpatterns = [
    path('multi/', views.score_segment, name='modelsimplemulti'),
    path('chosse-model/', views.dataset, name='dataset'),
    path('get-resultat/', views.test_dataset),
    path('js-to-py/', views.jstopy),
    path('temperature-actuelle/', views.temperatureactuel, name='modelsimpleactuel'),
    path('temperature-autre-ville/', views.changemodelresultat),
    path('temperature-actuelle-autre-ville/', views.temperatureactuelville),
    path('dataset-actuelle/', views.datasetactuelle, name='datasetactuelle'),
    path('noms-villes/', views.dataset_mobil),
]