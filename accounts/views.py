from django.shortcuts import render

from django.contrib.auth.decorators import login_required
from adminApi.models import Model, Parametre, Dataset
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

import tensorflow as tf
from tensorflow.keras import backend
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import math
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr

from adminApi.views import affectation_parametres_dataset,charger_opt,series_to_supervised,x_y_split_multi,x_y_split_simple
from dateutil.relativedelta import relativedelta


@login_required
def profil(request):
    return render(request, 'profile/changer_dataset.html', {})



@login_required
def adapter_model(request):
    models = Dataset.objects.all()
    return render(request, 'profile/adapter_model.html', {'models':models})

@login_required
@csrf_exempt
def get_parametre_model(request):
    id_dataset = request.POST['modelchoisi']
    dataset = Dataset.objects.get(id=id_dataset)
    model = Model.objects.get(dataset=dataset)
    parametres = Parametre.objects.get(model=model)
    data = {
        'id_dataset':id_dataset,
        'id_model':model.id,
        'window':model.window,
        'batch_size':parametres.batch_size,
        'learning_rate':parametres.learning_rate,
        'epochs':parametres.epochs,
        'optimizer':parametres.optimizer,
    }
    return JsonResponse(data)

@login_required
def transfer_learning(request):
    idataset = request.POST.get('idataset')
    idmodel = request.POST.get('idmodel')
    print('idmodel ',idmodel,'idataset',idataset)
    dataset_obj = Dataset.objects.get(id=idataset)
    model_obj = Model.objects.get(id=idmodel)
    type_mod,window,batch_size,optimizer,learning_rate,epochs = affectation_parametres_dataset(request)
    type_mod = model_obj.type_mod
    opt = charger_opt(optimizer,learning_rate)
    dataset = pd.read_csv("C:/Users/Rania/Desktop/Api/src/"+dataset_obj.dataset.url,index_col=0,header=0)
    dataset.index = pd.to_datetime(dataset.index, format='%Y-%m-%d %H:%M:%S')
    model =  load_model("C:/Users/Rania/Desktop/Api/src"+model_obj.model.url, compile=False)

    model.compile(loss='mae',
                optimizer=opt)

    values = dataset.values
    index = dataset.index
    normalized_y = MinMaxScaler(feature_range=(-1,1))
    y_norm = values[:,0]
    y_norm = y_norm.reshape(-1,1)
    y_norm = normalized_y.fit_transform(y_norm)
    y_normaliz = pd.DataFrame(y_norm,index=dataset.index, columns=['T (degC)'])
    

    if type_mod == 'simple':
        print('type_mod ani section simple',type_mod)
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
        print('type_mod ani section multi',type_mod)
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
    nouveau_model = model.save('media/models/'+model_obj.titre.replace(' ','_')+'.h5')
    Model.objects.filter(id=request.POST['idmodel']).update(window=window,model='models/'+model_obj.titre.replace(' ','_')+'.h5')
    Parametre.objects.filter(model=model_obj).update(batch_size=batch_size,optimizer=optimizer,learning_rate=learning_rate,epochs=epochs)
    
    try:
        lstm_valid_pred = model.predict(X_valid_reformer)
        MAE = mean_absolute_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average')
        MSE = mean_squared_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average')
        RMSE = math.sqrt(mean_squared_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average'))
        VAF = explained_variance_score(Y_valid, lstm_valid_pred, multioutput='uniform_average')
        corr, _ = pearsonr(Y_valid.ravel(), lstm_valid_pred.ravel())
        print('MAE === ',MAE)
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
    except Exception as error:
        print("error ===== ",str(error))
        data = {'error':str(error)}

    
    #return render(request, 'models.html', {})
    return JsonResponse(data)
