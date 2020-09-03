from django.db import models

# Create your models here.



class Prediction(models.Model):

    temperature = models.FloatField()
    date = models.DateTimeField()


    
