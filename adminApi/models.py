from django.db import models

# Create your models here.

def charger_dataset(instance,filename):
    name , extension = filename.split('.')
    return "dataset/%s.%s"%(instance.titre.replace(' ','_'),extension)


def charger_model(instance,filename):
    name , extension = filename.split('.')
    return "%s_%s.%s"%(instance.titre,instance.id,extension)


class Dataset(models.Model):
    titre = models.CharField(max_length=50)
    format_date = models.CharField(max_length=20)
    ordre = models.IntegerField(default=0)
    dataset = models.FileField(upload_to=charger_dataset)


class Model(models.Model):
    titre = models.CharField(max_length=100)
    model = models.FileField(upload_to=charger_model)
    window = models.IntegerField(default=72)
    type_mod = models.CharField(max_length=20, default='simple')
    dataset = models.ForeignKey('Dataset', on_delete=models.CASCADE)


class Parametre(models.Model):
    model = models.ForeignKey('Model', on_delete=models.CASCADE)
    batch_size = models.IntegerField()
    optimizer = models.CharField(max_length=10)
    learning_rate = models.FloatField()
    epochs = models.IntegerField()

      

