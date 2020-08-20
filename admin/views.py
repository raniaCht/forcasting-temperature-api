from django.shortcuts import render

# Create your views here.

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import backend
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from ModelsSimpleMulti.views import series_to_supervised
from keras.optimizers import SGD

def evalution_admin(request):
    model =  load_model('./models/model_simple_predire_temperature.h5', compile=False)

    # Adjust learning rate based on number of GPUs.
    opt = SGD(learning_rate=0.001)

    # Add Horovod Distributed Optimizer.
    #opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='mae',
                optimizer=opt)


    dataset = pd.read_csv('C:/Users/Rania/Desktop/Api/src/ModelsSimpleMulti/out.csv',index_col=0,header=0,error_bad_lines=False)
    

    print("**********************ani hna 1", dataset.shape)
    dataset.index = pd.to_datetime(dataset.index, format="%Y-%m-%d %H:%M:%S")
    values = dataset.values
    normalized_y = MinMaxScaler(feature_range=(-1,1))
    y_norm = values #.reshape(-1,1)
    print(y_norm.shape)
    y_norm = normalized_y.fit_transform(y_norm)
    
    y_norm = pd.DataFrame(y_norm,index=dataset.index, columns=['T (degC)'])
    
    series_multi = series_to_supervised(y_norm, window=72, lag=24, simple=False, single=False)
    print(series_multi.shape)
    # features = [('T (degC)(t+%d)' % (i)) for i in range(1, 25)]
    # y_multi = series_multi[features]
    # series_multi.drop(features, axis=1, inplace=True)
    #serie_multi = series_multi.loc['2015-01-04 01:00:00':'2017-01-01 00:00:00']
    
    X_valid_multi = np.array(series_multi)
    print(X_valid_multi.shape)
    X_valid_multi_reformer = X_valid_multi.reshape(X_valid_multi.shape[0], 73, 1) #[[-8.05,-8.88,-8.81,-9.05,-9.63,-9.67,-9.17,-8.1,-7.66,-7.04,-7.41,-6.87,-5.89,-5.94,-5.69,-5.4,-5.37,-5.25,-5.11,-4.9,-4.8,-4.5,-4.47,-4.54,-4.44,-4.29,-4.45,-4.58,-4.96,-4.43,-4.28,-4.33,-4.13,-3.93,-3.62,-3.12,-2.53,-2.56,-2.12,-2.76,-2.88,-3.07,-3.34,-3.3,-3.49,-4.02,-4.38,-4.71,-5.28,-6.23,-6.13,-6.21,-7.02,-8.2,-8.48,-9.28,-9.46,-8.53,-7.87,-5.96,-2.82,-2.15,-0.82,-2.8,-4.69,-4.08,-3.78,-3.84,-3.15,-2.76,-2.58,-1.9,-1.42]] 
    
    #X_valid_multi_reformer = np.array(X_valid_multi_reformer)
    #X_valid_multi_reformer = X_valid_multi_reformer.reshape(X_valid_multi_reformer.shape[0], 73, 1)

    try:
        predictions = model.predict(X_valid_multi_reformer)
        predictions = normalized_y.inverse_transform(predictions[-1:])
    except Exception as error:
        predictions = str(error)
    
    
    return render(request, 'models.html', {'predictions': predictions})
