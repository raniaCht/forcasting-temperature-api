from rest_framework import serializers
from .models import Prediction



class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['temperature', 'date']