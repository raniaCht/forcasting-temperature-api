from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import Api.settings
from keras.models import load_model
import requests
from django.http import JsonResponse

from keras.models import load_model

from adminApi.models import Model, Dataset

from .models import Prediction
from .serializers import PredictionSerializer
# Create your views here.


def series_to_supervised(data, window=1, lag=1, dropnan=True, simple=True, single=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # if simple == False:
    #     # Target timestep (t=lag)
    #     if single == True:
    #         cols.append(data.shift(-lag))
    #         names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    #     if single == False:
    #         for j in range(1, lag+1, 1):
    #             cols.append(data.shift(-j))
    #             names += [('%s(t+%d)' % (col, j)) for col in data.columns]

    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.index = data.index
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


@api_view(['GET'])
def score_segment(request):
    dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/out.csv',index_col=0,header=0,error_bad_lines=False)
    dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
    indexactuel = datetime.now()
    dateactuel = datetime(indexactuel.year, indexactuel.month, indexactuel.day,indexactuel.hour,00,00)
    temperature_current = dataset.values[dataset.index == dateactuel]
    model = Api.settings.gModelObjs['multi_model']
    while (len(temperature_current) == 0):
    
        values = dataset.values[-73:]
        index = dataset.index[-73:]
        normalized_y = MinMaxScaler(feature_range=(-1,1))
        y_norm = values
        print(y_norm.shape)
        y_norm = normalized_y.fit_transform(y_norm)
        
        y_norm = pd.DataFrame(y_norm,index=index, columns=['T (degC)'])
        
        series_multi = series_to_supervised(y_norm, window=72, lag=24, simple=False, single=False)
        
        X_valid_multi = np.array(series_multi)
        print(X_valid_multi.shape)
        X_valid_multi_reformer = X_valid_multi.reshape(X_valid_multi.shape[0], 73, 1) 

        try:
            predictions = model.predict(X_valid_multi_reformer)
            predictions = normalized_y.inverse_transform(predictions[-1:])
            predictionBD = save_data(predictions[-1:],dataset.index.max())

            dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/out.csv',index_col=0,header=0,error_bad_lines=False)
            dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
            temperature_current = dataset.values[dataset.index == dateactuel]
            print("temperature actuel ======= ", temperature_current)
        except Exception as error:
            return Response({'message':str(error)})
        
    else:
        predictionBD = list()
        indexactuel = datetime.now()
        dateactuel = datetime(indexactuel.year, indexactuel.month, indexactuel.day,1,00,00)
        datefinal = datetime(indexactuel.year, indexactuel.month, indexactuel.day,00,00,00)
        datefinal += timedelta(days=1)

        while dateactuel <= datefinal:
            
            temperature_current = dataset.values[dataset.index == dateactuel]
            prediction = Prediction(temperature=dataset.values[dataset.index == dateactuel][0][0],date=dateactuel)
            predictionBD.append(prediction)
            dateactuel += timedelta(hours=1)
    
    data = PredictionSerializer(predictionBD, many=True).data
    return Response(data)

# @api_view(['GET'])
# def score_segment(request):
#     # We no longer need to load the model here. It's already preloaded.
#     # graph, model = _load_model_from_path('path_to_keras_model')
#     model = Api.settings.gModelObjs['multi_model'] # if model_1 is used
#     print("**********************ani hna 1")
#     dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/out.csv',index_col=0,header=0,error_bad_lines=False)
#     print("**********************ani hna 1", dataset.shape)
#     dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
#     values = dataset.values[-73:]
#     index = dataset.index[-73:]
#     normalized_y = MinMaxScaler(feature_range=(-1,1))
#     y_norm = values #.reshape(-1,1)
#     print(y_norm.shape)
#     y_norm = normalized_y.fit_transform(y_norm)
    
#     y_norm = pd.DataFrame(y_norm,index=index, columns=['T (degC)'])
    
#     series_multi = series_to_supervised(y_norm, window=72, lag=24, simple=False, single=False)
#     print(series_multi.shape)
#     # features = [('T (degC)(t+%d)' % (i)) for i in range(1, 25)]
#     # y_multi = series_multi[features]
#     # series_multi.drop(features, axis=1, inplace=True)
#     #serie_multi = series_multi.loc['2015-01-04 01:00:00':'2017-01-01 00:00:00']
    
#     X_valid_multi = np.array(series_multi)
#     print(X_valid_multi.shape)
#     X_valid_multi_reformer = X_valid_multi.reshape(X_valid_multi.shape[0], 73, 1) #[[-8.05,-8.88,-8.81,-9.05,-9.63,-9.67,-9.17,-8.1,-7.66,-7.04,-7.41,-6.87,-5.89,-5.94,-5.69,-5.4,-5.37,-5.25,-5.11,-4.9,-4.8,-4.5,-4.47,-4.54,-4.44,-4.29,-4.45,-4.58,-4.96,-4.43,-4.28,-4.33,-4.13,-3.93,-3.62,-3.12,-2.53,-2.56,-2.12,-2.76,-2.88,-3.07,-3.34,-3.3,-3.49,-4.02,-4.38,-4.71,-5.28,-6.23,-6.13,-6.21,-7.02,-8.2,-8.48,-9.28,-9.46,-8.53,-7.87,-5.96,-2.82,-2.15,-0.82,-2.8,-4.69,-4.08,-3.78,-3.84,-3.15,-2.76,-2.58,-1.9,-1.42]] 
    
#     #X_valid_multi_reformer = np.array(X_valid_multi_reformer)
#     #X_valid_multi_reformer = X_valid_multi_reformer.reshape(X_valid_multi_reformer.shape[0], 73, 1)

#     try:
#         predictions = model.predict(X_valid_multi_reformer)
#         predictions = normalized_y.inverse_transform(predictions[-1:])
#         data = PredictionSerializer(save_data(predictions[-1:],dataset.index.max()),many=True).data
#     except Exception as error:
#         return Response({'message':str(error)})
#     return Response(data)


def save_data(predictions, max_date):
    print("**********************ani hna 2")
    date = datetime(max_date.year,max_date.month,max_date.day,max_date.hour,00,00)
    predictionsBD = list()
    out = open('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/out.csv', 'a')
    for row in predictions:
        for column in row:
            
            date += timedelta(hours=1)
            out.write('{},'.format(date))
            out.write('{:.2f}'.format(column))
            out.write('\n')
            prediction = Prediction(temperature=column,date=date)
            predictionsBD.append(prediction)
            prediction.save()
    out.close()
    return predictionsBD


def charger_model(path):
    model = load_model(path)
    return model



@api_view(['GET'])
def test_dataset(request):
    # We no longer need to load the model here. It's already preloaded.
    # graph, model = _load_model_from_path('path_to_keras_model')
    model_obj = Model.objects.get(id=request.GET['id_model'])
    print("path *** model ******",model_obj.model)
    model = charger_model("C:/Users/Rania/Desktop/Api/src/media/models/matalan.h5")
    #model = Api.settings.gModelObjs['multi_model'] # if model_1 is used
    
    print("**********************ani hna 1")
    dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/out.csv',index_col=0,header=0,error_bad_lines=False)
    print("**********************ani hna 1", dataset.shape)
    dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
    values = dataset.values[-25:]
    index = dataset.index[-25:]
    normalized_y = MinMaxScaler(feature_range=(-1,1))
    y_norm = values #.reshape(-1,1)
    print(y_norm.shape)
    y_norm = normalized_y.fit_transform(y_norm)
    print(y_norm.shape)
    y_norm = pd.DataFrame(y_norm,index=index, columns=['T (degC)'])
    print('values ===',y_norm.values)
    series_multi = series_to_supervised(y_norm, window=24)
    print(type(series_multi))
    series_multi.drop(['T (degC)(t)'], axis=1, inplace=True)
    print(series_multi.shape)
    # features = [('T (degC)(t+%d)' % (i)) for i in range(1, 25)]
    # y_multi = series_multi[features]
    # series_multi.drop(features, axis=1, inplace=True)
    #serie_multi = series_multi.loc['2015-01-04 01:00:00':'2017-01-01 00:00:00']
    print('values ===',series_multi.values)
    X_valid_multi = series_multi.values
    print(X_valid_multi.shape[0])
    X_valid_multi_reformer = X_valid_multi.reshape(X_valid_multi.shape[0], 24, 1) #[[-8.05,-8.88,-8.81,-9.05,-9.63,-9.67,-9.17,-8.1,-7.66,-7.04,-7.41,-6.87,-5.89,-5.94,-5.69,-5.4,-5.37,-5.25,-5.11,-4.9,-4.8,-4.5,-4.47,-4.54,-4.44,-4.29,-4.45,-4.58,-4.96,-4.43,-4.28,-4.33,-4.13,-3.93,-3.62,-3.12,-2.53,-2.56,-2.12,-2.76,-2.88,-3.07,-3.34,-3.3,-3.49,-4.02,-4.38,-4.71,-5.28,-6.23,-6.13,-6.21,-7.02,-8.2,-8.48,-9.28,-9.46,-8.53,-7.87,-5.96,-2.82,-2.15,-0.82,-2.8,-4.69,-4.08,-3.78,-3.84,-3.15,-2.76,-2.58,-1.9,-1.42]] 
    
    #X_valid_multi_reformer = np.array(X_valid_multi_reformer)
    #X_valid_multi_reformer = X_valid_multi_reformer.reshape(X_valid_multi_reformer.shape[0], 73, 1)

    try:
        predictions = model.predict(X_valid_multi_reformer)
        predictions = normalized_y.inverse_transform(predictions[-1:])
        print('prediction == ',predictions)
        #data = PredictionSerializer(save_data(predictions[-1:],dataset.index.max()),many=True).data
    except Exception as error:
        return Response({'message':str(error)})
    return Response({'predictions':predictions})



def jstopy(request):
    if request.method == 'POST':
        list_models = Model.objects.get(type = request.POST['type_mod'])
        models = '<option value="" selected hidden disabled>---</option>'
        for model in list_models:
            models += '<option value="'+model.id+'">'+model.titre+'</option>'
        print('ani hna ======')
        return models

@api_view(['GET'])
def temperatureactuel(request):
    dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/out.csv',index_col=0,header=0,error_bad_lines=False)
    dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
    indexactuel = datetime.now()
    dateactuel = datetime(indexactuel.year, indexactuel.month, indexactuel.day,indexactuel.hour,00,00)
    temperature_current = dataset.values[dataset.index == dateactuel]
    print("temperature actuel ======= ", temperature_current)
    while (len(temperature_current) == 0):
        r = requests.get('http://127.0.0.1:8000/models/multi/')
        print("statut of request :", r.status_code)
        dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/out.csv',index_col=0,header=0,error_bad_lines=False)
        dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
        temperature_current = dataset.values[dataset.index == dateactuel]
        print("temperature actuel ======= ", temperature_current)
    return Response([{ 'temperature_current' :temperature_current[0][0]},])



def dataset(request):
    list_models = Dataset.objects.all()
    return render(request, 'test_dataset.html', {'list_models':list_models})

@api_view(['GET'])
def changemodelresultat(request):
    modelchoisi = request.GET.get('modelchoisi')

    print('ani hnaaaa ====',modelchoisi)
    dataset_obj = Dataset.objects.get(id=modelchoisi)
    
    model = Model.objects.get(dataset=dataset_obj)
    print("ani hnaaaa=====",model.titre)
    model_load = load_model('C:/Users/Rania/Desktop/Api/src'+model.model.url)
    dataset = pd.read_csv("C:/Users/Rania/Desktop/Api/src/"+dataset_obj.dataset.url,index_col=0,header=0)
    dataset.index = pd.to_datetime(dataset.index, format='%Y-%m-%d %H:%M:%S')
    indexactuel = datetime.now()
    dateactuel = datetime(indexactuel.year, indexactuel.month, indexactuel.day,00,00,00) #indexactuel.strftime(dataset_obj.format_date) 
    dateactuel += timedelta(days=1)
    print("dateactuel dateactuel",dateactuel)
    temperature_current = dataset.values[dataset.index == dateactuel]
    print("temperature_current temperature_current",temperature_current)
    while (len(temperature_current) == 0):
        i_start = model.window+1
        values = dataset.values[-i_start:]
        index = dataset.index[-i_start:]
        normalized_y = MinMaxScaler(feature_range=(-1,1))
        y_norm = values
        print("shape window ",y_norm.shape)
        y_norm = normalized_y.fit_transform(y_norm)
        
        y_norm = pd.DataFrame(y_norm,index=index, columns=['T (degC)'])
        
        series_multi = series_to_supervised(y_norm, window=model.window, lag=24, simple=False, single=False)
        
        X_valid_multi = np.array(series_multi)
        print(X_valid_multi.shape)
        X_valid_multi_reformer = X_valid_multi.reshape(X_valid_multi.shape[0], model.window+1, 1) 

        try:
            predictions = model_load.predict(X_valid_multi_reformer)
            print("prediction =====",predictions)
            predictions = normalized_y.inverse_transform(predictions[-1:])
            predictionBD = save_data_ville(predictions[-1:],dataset.index.max(),dataset_obj.dataset.url)

            dataset = pd.read_csv("C:/Users/Rania/Desktop/Api/src/"+dataset_obj.dataset.url,index_col=0,header=0)
            dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
            temperature_current = dataset.values[dataset.index == dateactuel]
            print("temperature actuel ======= ", temperature_current)
        except Exception as error:
            return Response({'message':str(error)})
        
    else:
        predictionBD = list()
        indexactuel = datetime.now()
        dateactuel = datetime(indexactuel.year, indexactuel.month, indexactuel.day,1,00,00)
        datefinal = datetime(indexactuel.year, indexactuel.month, indexactuel.day,00,00,00)
        datefinal += timedelta(days=1)
        print('datefinal ',datefinal)
        dataset = pd.read_csv("C:/Users/Rania/Desktop/Api/src/"+dataset_obj.dataset.url,index_col=0,header=0)
        dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
        print("shape dataset === ",dataset.shape)
        while dateactuel <= datefinal:
            print("dataset",dataset)
            print('dateactuel',dateactuel)
            print("dataset.values",dataset.values)
            print('dataset.index',dataset.index)
            print("dataset.index == dateactuel ",dataset.index == dateactuel)
            temperature_current = dataset.values[dataset.index == dateactuel]
            print(temperature_current.shape)
            prediction = Prediction(temperature=temperature_current[0],date=dateactuel)
            predictionBD.append(prediction)
            dateactuel += timedelta(hours=1)
    
    data = PredictionSerializer(predictionBD, many=True).data
    return Response(data)


def save_data_ville(predictions, max_date,url):
    print("**********************ani hna 2")
    date = datetime(max_date.year,max_date.month,max_date.day,max_date.hour,00,00)
    predictionsBD = list()
    out = open('C:/Users/Rania/Desktop/Api/src/'+url, 'a')
    for row in predictions:
        for column in row:
            
            date += timedelta(hours=1)
            out.write('{},'.format(date))
            out.write('{:.2f}'.format(column))
            out.write('\n')
            prediction = Prediction(temperature=column,date=date)
            predictionsBD.append(prediction)
            # prediction.save()
    out.close()
    return predictionsBD