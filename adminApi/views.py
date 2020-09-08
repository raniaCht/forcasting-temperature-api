from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
# Create your views here.
import math

import tensorflow as tf
from tensorflow.keras import backend
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr
from .models import Dataset, Model , Parametre

from dateutil.relativedelta import relativedelta


parameter_entrainement = {
    'simple':{
        'window':24,
        'batch_size':128,
        'optimizer':'Adam',
        'learning_rate':0.0001,
        'epochs':80,
    },
    'multi':{
        'window':72,
        'batch_size':256,
        'optimizer':'Adam',
        'learning_rate':0.001,
        'epochs':80,
    },
}


def series_to_supervised(data, window=1, lag=1, dropnan=True, simple=True, single=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)

    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    if simple == False:
        # Target timestep (t=lag)
        if single == True:
            cols.append(data.shift(-lag))
            names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
        if single == False:
            for j in range(1, lag+1, 1):
                cols.append(data.shift(-j))
                names += [('%s(t+%d)' % (col, j)) for col in data.columns]

    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.index = data.index
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def home(request):
    return render(request, 'models.html', {})

def loader_model(type):
    if type == 'simple':
        model =  load_model('./models/model_simple_predire_temperature.h5', compile=False)
    else:
        model =  load_model('./models/model_multi_predire_temperature.h5', compile=False)
    return model


def affectation_parametres(request):
    if request.GET['type'] != '':
        type_mod = request.GET['type']
    else:
        type_mod = 'simple'

    if request.GET['window'] != '':
        window = int(request.GET['window'])
    else:
        window = int(parameter_entrainement[type_mod]['window'])

    if request.GET['batch_size'] != '':
        batch_size = int(request.GET['batch_size'])
    else:
        batch_size = int(parameter_entrainement[type_mod]['batch_size'])

    if request.GET['optimizer'] != '':
        optimizer = request.GET['optimizer']
    else:
        optimizer = parameter_entrainement[type_mod]['optimizer']

    if request.GET['learning_rate'] != '':
        learning_rate = float(request.GET['learning_rate'])
    else:
        learning_rate = float(parameter_entrainement[type_mod]['learning_rate'])

    if request.GET['epochs'] != '':
        epochs = int(request.GET['epochs'])
    else:
        epochs = int(parameter_entrainement[type_mod]['epochs'])

    return type_mod,window,batch_size,optimizer,learning_rate,epochs


def charger_opt(optimizer,learning_rate):
    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)

    return opt



def evalution_admin(request):

    type_mod,window,batch_size,optimizer,learning_rate,epochs = affectation_parametres(request)
    model =  loader_model(type_mod)

    opt = charger_opt(optimizer,learning_rate)


    model.compile(loss='mae',
                optimizer=opt)


    #dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/climate_hour.csv',index_col=0,header=0)
    dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/temperature_data.csv',index_col=0,header=0)

    print("**********************ani hna 1", dataset.shape)
    dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
    values = dataset.values

    normalized_y = MinMaxScaler(feature_range=(-1,1))
    y_norm = values[:,0]
    y_norm = y_norm.reshape(-1,1)
    y_norm = normalized_y.fit_transform(y_norm)
    y_normaliz = pd.DataFrame(y_norm,index=dataset.index, columns=['T (degC)'])

    if type_mod == 'simple':
        series = series_to_supervised(y_normaliz, window=window)
        y = series['T (degC)(t)']
        series.drop(['T (degC)(t)'], axis=1, inplace=True)
        timesteps = window
        X_train_reformer,Y_train,X_valid_reformer,Y_valid = x_y_split_simple(series,y,timesteps)
    else:
        series = series_to_supervised(y_normaliz, window=window, lag=24, simple=False, single=False)
        features = [('T (degC)(t+%d)' % (i)) for i in range(1, 25)]
        y = series[features]
        series.drop(features, axis=1, inplace=True)
        timesteps = window + 1
        X_train_reformer,Y_train,X_valid_reformer,Y_valid = x_y_split_multi(series,y,timesteps)

    lstm_history = model.fit(X_train_reformer,Y_train,validation_data=(X_valid_reformer,Y_valid), epochs=epochs, batch_size=batch_size, verbose=1)

    try:
        lstm_valid_pred = model.predict(X_valid_reformer)
        MAE = mean_absolute_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average')
    except Exception as error:
        MAE = str(error)

    data = {
        'label' : [('%d' % (i)) for i in range(1, epochs +1)],
        'label_epoch' : [('Epoch %d' % (i)) for i in range(1, epochs+1)],
        'LossofFitting' : lstm_history.history['loss'],
        'LossofValidation' : lstm_history.history['val_loss'],
    }
    #return render(request, 'models.html', {})
    return JsonResponse(data)


def x_y_split_simple(series,y,timesteps):

    # X_train = series.loc['2009-01-02 01:00:00':'2015-01-02 00:00:00']
    # X_valid = series.loc['2015-01-02 01:00:00':'2017-01-01 00:00:00']
    # Y_train = y.loc['2009-01-02 01:00:00':'2015-01-02 00:00:00']
    # Y_valid = y.loc['2015-01-02 01:00:00':'2017-01-01 00:00:00']


    X_train = series.loc['2009-01-02 01:00:00':'2017-01-01 01:00:00']
    X_valid = series.loc['2017-01-01 01:00:00':'2020-01-01 00:00:00']
    Y_train = y.loc['2009-01-02 01:00:00':'2017-01-01 01:00:00']
    Y_valid = y.loc['2017-01-01 01:00:00':'2020-01-01 00:00:00']

    X_train = X_train.values
    Y_train = Y_train.values
    X_valid = X_valid.values
    Y_valid = Y_valid.values
    Y_train = Y_train.reshape(-1,1)
    Y_valid = Y_valid.reshape(-1,1)

    ndim = 1
    X_train_reformer = X_train.reshape(X_train.shape[0],timesteps,ndim)
    X_valid_reformer = X_valid.reshape(X_valid.shape[0],timesteps,ndim)
    Y_train = Y_train.reshape(Y_train.shape[0],)
    Y_valid = Y_valid.reshape(Y_valid.shape[0],)

    return X_train_reformer,Y_train,X_valid_reformer,Y_valid

def x_y_split_multi(series,y,timesteps):
    series_multi  = series.sort_values('Date Time')
    # X_train_multi = series.loc['2009-01-04 01:00:00':'2015-01-04 00:00:00']
    # X_valid_multi = series.loc['2015-01-04 01:00:00':'2017-01-01 00:00:00']
    # Y_train_multi = y.loc['2009-01-04 01:00:00':'2015-01-04 00:00:00']
    # Y_valid_multi = y.loc['2015-01-04 01:00:00':'2017-01-01 00:00:00']

    X_train_multi = series.loc['2009-01-04 01:00:00':'2017-01-04 00:00:00']
    X_valid_multi = series.loc['2017-01-04 01:00:00':'2020-01-01 00:00:00']
    Y_train_multi = y.loc['2009-01-04 01:00:00':'2017-01-04 00:00:00']
    Y_valid_multi = y.loc['2017-01-04 01:00:00':'2020-01-01 00:00:00']

    X_train_multi = X_train_multi.values
    Y_train_multi = Y_train_multi.values
    X_valid_multi = X_valid_multi.values
    Y_valid_multi = Y_valid_multi.values
    Y_train_multi = Y_train_multi.reshape(-1,24)
    Y_valid_multi = Y_valid_multi.reshape(-1,24)

    ndim = 1
    X_train_multi_reformer = X_train_multi.reshape(X_train_multi.shape[0],timesteps,ndim)
    X_valid_multi_reformer = X_valid_multi.reshape(X_valid_multi.shape[0],timesteps,ndim)
    Y_train_multi = Y_train_multi.reshape(Y_train_multi.shape[0],24)
    Y_valid_multi = Y_valid_multi.reshape(Y_valid_multi.shape[0],24)

    return X_train_multi_reformer,Y_train_multi,X_valid_multi_reformer,Y_valid_multi


# def entrainerfenetre(request):
#     model = loader_model(request.GET['type'])
#     fenetre = int(request.GET['fenetre'])

#     opt = Adam(learning_rate=0.001)


#     model.compile(loss='mae',
#                 optimizer=opt)
#     dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/climate_hour.csv',index_col=0,header=0)
#     dataset.index = pd.to_datetime(dataset.index, format="%d.%m.%Y %H:%M:%S")
#     values = dataset.values

#     normalized_y = MinMaxScaler(feature_range=(-1,1))
#     y_norm = values[:,1]
#     y_norm = y_norm.reshape(-1,1)
#     y_norm = normalized_y.fit_transform(y_norm)
#     y_normaliz = pd.DataFrame(y_norm,index=dataset.index, columns=['T (degC)'])

#     series = series_to_supervised(y_normaliz, window=fenetre)

#     y = series['T (degC)(t)']
#     y.shape
#     series.drop(['T (degC)(t)'], axis=1, inplace=True)

#     X_train = series.loc['2009-01-02 01:00:00':'2015-01-02 00:00:00']
#     X_valid = series.loc['2015-01-02 01:00:00':'2017-01-01 00:00:00']
#     Y_train = y.loc['2009-01-02 01:00:00':'2015-01-02 00:00:00']
#     Y_valid = y.loc['2015-01-02 01:00:00':'2017-01-01 00:00:00']


#     X_train = X_train.values
#     Y_train = Y_train.values
#     X_valid = X_valid.values
#     Y_valid = Y_valid.values
#     Y_train = Y_train.reshape(-1,1)
#     Y_valid = Y_valid.reshape(-1,1)

#     timesteps = fenetre
#     ndim = 1
#     X_train_reformer = X_train.reshape(X_train.shape[0],timesteps,ndim)
#     X_valid_reformer = X_valid.reshape(X_valid.shape[0],timesteps,ndim)
#     Y_train = Y_train.reshape(Y_train.shape[0],)
#     Y_valid = Y_valid.reshape(Y_valid.shape[0],)

#     epoch = 5
#     lstm_history = model.fit(X_train_reformer,Y_train,validation_data=(X_valid_reformer,Y_valid), epochs=epoch, batch_size=128, shuffle=False, verbose=1)

#     try:
#         lstm_valid_pred = model.predict(X_valid_reformer)
#         MAE = mean_absolute_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average')
#     except Exception as error:
#         MAE = str(error)

#     data = {
#         'label' : [('%d' % (i)) for i in range(1, epoch+1)],
#         'label_epoch' : [('Epoch %d' % (i)) for i in range(1, epoch+1)],
#         'LossofFitting' : lstm_history.history['loss'],
#         'LossofValidation' : lstm_history.history['val_loss'],
#     }
#     #return render(request, 'models.html', {})
#     return JsonResponse(data)


def index(request):
    return render(request, 'index.html', {})


def dataset(request):
    return render(request, 'dataset.html', {})


def set_dataset(request):

    ordre = request.POST['ordre']
    print('ordre ',ordre)
    type_mod,window,batch_size,optimizer,learning_rate,epochs = affectation_parametres_dataset(request)
    model =  loader_model(type_mod)

    opt = charger_opt(optimizer,learning_rate)


    model.compile(loss='mae',
                optimizer=opt)

    dataset_obj = Dataset(titre=request.POST['titre'], format_date=request.POST['format'], dataset=request.FILES['dataset'])
    dataset_obj.save()
    dataset = pd.read_csv("C:/Users/Rania/Desktop/Api/src/"+dataset_obj.dataset.url,index_col=0,header=0)
    print(dataset.values.shape[0])
    dataset.index = pd.to_datetime(dataset.index, format=dataset_obj.format_date)
    values = dataset.values
    index = dataset.index
    normalized_y = MinMaxScaler(feature_range=(-1,1))
    y_norm = values[:,int(ordre)]
    y_norm = y_norm.reshape(-1,1)
    y_nnormaliz = pd.DataFrame(y_norm,index=dataset.index, columns=['T (degC)'])
    y_norm = normalized_y.fit_transform(y_norm)
    y_normaliz = pd.DataFrame(y_norm,index=dataset.index, columns=['T (degC)'])
    
    y_nnormaliz.to_csv("C:/Users/Rania/Desktop/Api/src/"+dataset_obj.dataset.url)

    if type_mod == 'simple':
        series = series_to_supervised(y_normaliz, window=window)
        total = series.values.shape[0]
        y = series['T (degC)(t)']
        series.drop(['T (degC)(t)'], axis=1, inplace=True)
        timesteps = window
        X_train_reformer,Y_train,X_valid_reformer,Y_valid = x_y_split_simple(series,y,timesteps)

        min_date = min(index)
        max_date = max(index)
        start_validation = max_date - relativedelta(years=2)

        X_train = series.loc[min_date:start_validation]
        X_valid = series.loc[start_validation:max_date]
        Y_train = y.loc[min_date:start_validation]
        Y_valid = y.loc[start_validation:max_date]
        nbr_train = X_train.shape[0]
        nbr_valid = X_valid.shape[0]

        X_train = X_train.values
        Y_train = Y_train.values
        X_valid = X_valid.values
        Y_valid = Y_valid.values
        Y_train = Y_train.reshape(-1,1)
        Y_valid = Y_valid.reshape(-1,1)

        ndim = 1
        X_train_reformer = X_train.reshape(X_train.shape[0],timesteps,ndim)
        X_valid_reformer = X_valid.reshape(X_valid.shape[0],timesteps,ndim)
        Y_train = Y_train.reshape(Y_train.shape[0],)
        Y_valid = Y_valid.reshape(Y_valid.shape[0],)
    else:
        series = series_to_supervised(y_normaliz, window=window, lag=24, simple=False, single=False)
        total = series.values.shape[0]
        features = [('T (degC)(t+%d)' % (i)) for i in range(1, 25)]
        y = series[features]
        series.drop(features, axis=1, inplace=True)
        timesteps = window + 1

        min_date = min(index)
        max_date = max(index)
        start_validation = max_date - relativedelta(years=2)

        X_train_multi = series.loc[min_date:start_validation]
        X_valid_multi = series.loc[start_validation:max_date]
        Y_train_multi = y.loc[min_date:start_validation]
        Y_valid_multi = y.loc[start_validation:max_date]

        nbr_train = X_train_multi.shape[0]
        nbr_valid = X_valid_multi.shape[0]

        X_train_multi = X_train_multi.values
        Y_train_multi = Y_train_multi.values
        X_valid_multi = X_valid_multi.values
        Y_valid_multi = Y_valid_multi.values
        Y_train_multi = Y_train_multi.reshape(-1,24)
        Y_valid_multi = Y_valid_multi.reshape(-1,24)

        ndim = 1
        X_train_reformer = X_train_multi.reshape(X_train_multi.shape[0],timesteps,ndim)
        X_valid_reformer = X_valid_multi.reshape(X_valid_multi.shape[0],timesteps,ndim)
        Y_train = Y_train_multi.reshape(Y_train_multi.shape[0],24)
        Y_valid = Y_valid_multi.reshape(Y_valid_multi.shape[0],24)


    lstm_history = model.fit(X_train_reformer,Y_train,validation_data=(X_valid_reformer,Y_valid), epochs=epochs, batch_size=batch_size, shuffle=False, verbose=1)
    print(lstm_history.history['loss'])
    nouveau_model = model.save('media/models/'+request.POST['titre_mod'].replace(' ','_')+'.h5')
    nouveau_model = Model(titre=request.POST['titre_mod'],window=window,dataset=dataset_obj,model='models/'+request.POST['titre_mod'].replace(' ','_')+'.h5',type_mod=type_mod)
    nouveau_model.save()
    parametre = Parametre(model=nouveau_model,batch_size=batch_size,optimizer=optimizer,learning_rate=learning_rate,epochs=epochs)
    parametre.save()

    print(nouveau_model.model)
    try:
        lstm_valid_pred = model.predict(X_valid_reformer)
        MAE = mean_absolute_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average')
        MSE = mean_squared_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average')
        RMSE = math.sqrt(mean_squared_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average'))
        VAF = explained_variance_score(Y_valid, lstm_valid_pred, multioutput='uniform_average')
        corr, _ = pearsonr(Y_valid.ravel(), lstm_valid_pred.ravel())
        print('MAE === ',MAE)
    except Exception as error:
        MAE = str(error)
        print("error ===== ",str(error))

    data = {
        'done':"done",
        'label' : [('%d' % (i)) for i in range(1, epochs +1)],
        'label_epoch' : [('Epoch %d' % (i)) for i in range(1, epochs+1)],
        'LossofFitting' : lstm_history.history['loss'],
        'LossofValidation' : lstm_history.history['val_loss'],
        'MAE':MAE,
        'MSE':MSE,
        'RMSE':RMSE,
        'VAF':VAF,
        'corr':corr,
        'total':total,
        'nbr_train':nbr_train,
        'nbr_valid':nbr_valid,
    }
    #return render(request, 'models.html', {})
    return JsonResponse(data)


def affectation_parametres_dataset(request):

    type_mod = request.POST.get('type' , 'simple')

    window = int(request.POST.get('window',parameter_entrainement[type_mod]['window']))

    batch_size = int(request.POST.get('batch_size',parameter_entrainement[type_mod]['batch_size']))

    optimizer = request.POST.get('optimizer',parameter_entrainement[type_mod]['optimizer'])

    learning_rate = float(request.POST.get('learning_rate',parameter_entrainement[type_mod]['learning_rate']))

    if request.POST['epochs'] != '':
        epochs = int(request.POST['epochs'])
    else:
        epochs = int(parameter_entrainement[type_mod]['epochs'])

    return type_mod,window,batch_size,optimizer,learning_rate,epochs



def visual_data(request):
    position = int(request.POST['ordre'])
    try:
        csv_file = request.FILES["dataset"]
        file_data = csv_file.read().decode("utf-8")
        data_time = []
        temperatures = []
        lines = file_data.split("\n")
        #loop over the lines and save them in db. If error , store as string and then display
        data_dict = {}
        print(len(lines))
        print(lines[4])
        for line in lines:
            try:
                # print(i)
                fields = line.split(",")
                data_time.append(fields[0])
                temperatures.append(fields[position+1])
            except Exception as e:
                print(e)

        data_dict["data_time"] = data_time
        data_dict["temparature"] = temperatures
    except Exception as e:
        data = {'error':''}

    return JsonResponse(data_dict)