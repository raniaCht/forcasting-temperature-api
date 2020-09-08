from rest_framework import serializers
from .models import Prediction
from adminApi.models import Dataset



class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['temperature', 'date']


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'titre']