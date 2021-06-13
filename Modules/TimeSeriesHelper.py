# -*- coding: utf-8 -*-
"""
Created on Sun May  2 10:57:56 2021

@author: Student
"""
import numpy as np
import tensorflow as tf

def split_sequence(sequences,to_predict, n_steps=1):
    """
    
    function that splits a dataset sequence into input data and
    labels
    
    splits the sequences to windows and true values.
    """
    X, Y = [], []
    for i in range(sequences.shape[0]):
        if(i + n_steps) >= sequences.shape[0]:
            break
          # Divide sequence between data (input) and labels (output)
        seq_X, seq_Y = sequences[i: i + n_steps],to_predict[i + n_steps]
                     
        X.append(seq_X)
        Y.append(seq_Y)
    return np.array(X), np.array(Y)


def single_multi_forecast(model,first_step,forecast_length=8,scaler = None):
    """
    Forecasts multi steps to the future from one window, using recursive multi-step forecasting.
    
    :param model: keras Sequential model. The model used to forecast.
    :param first_step: a numpy array. The window to use to start the recursive multi-step forecast
    :param forecast_length: an integer. The amount of timesteps to forecast
    
    """
    current_step = first_step
    future = []
    for i in range(forecast_length):
        prediction = model.predict(current_step.reshape((1,current_step.shape[0],current_step.shape[1]))) #get the next step
        future.append(prediction) #store the future prediction
        current_step = np.concatenate([current_step[1:],prediction]) # concatenate the new prediction to the 
    future = np.array(future)
    future = np.concatenate(future,axis=0)
    if scaler != None:
        future = scaler.inverse_transform(future.reshape(-1, 1)).reshape((forecast_length,))
    return future


def multi_forecast(model, data ,forecast_length=8):
    """
    Forecasts forecast_length timesteps into the future for each window and returns a list of numpy arrays that contain the forecasts.
    
    :param model: keras Sequential model. The model used to forecast.
    :param data: a numpy array. The array of the windows to use to forecast.
    :param forecast_length: an integer. The amount of timesteps to forecast for each window.
    
    """
    future_forecasts = []
    for i in range(data.shape[0]):
        current_step = data[i]
        future = []
        for j in range(forecast_length):
            forecast = model.predict(current_step.reshape(1,current_step.shape[0],current_step.shape[1]))
            future.append(forecast)
            current_step = np.concatenate([current_step[1:],forecast])
        future_forecasts.append(np.array(future))
    return future_forecasts
    

def naive_forecast(series,typ='last_next',window_size = None):
    """
    Uses naive forecast methodes to forecast the series and returnsa numpy array of the forecasts.
    
    :param series: a pandas Series object. The series to forecast.
    :param typ: 'last_next' or 'mean_average'. The type of the forecast to use. 
    :param window_size: an integer. The window_size for the mean average if used.
    :returns: an array with the forecasts.
    
    Note:
    mean_average - series[n] = the average of series[n-window_size:n]
    last_next - the forecast: series[n+1] = series[n]
    """
    if typ=='last_next':
        naive_prediction = series.shift(1).values
    elif typ=='mean_average':
        try:
            naive_prediction = series.rolling(window_size).mean().to_numpy().reshape(-1,1)[window_size-1:-window_size]
        except:
            raise Exception('Value Error: You didnt enter a window_size or the series is not valid')
    else:
        raise Exception('Value Error: You must enter either "mean_average" or "last_next" for the typ parameter.')
    return naive_prediction

def calculate_multi_forecast_errors(forecasts, trues, forecast_length=8):
    """
    Calculates the MAE and RMSE of the multi-step forecasts and returns a dictionary with the values.
    
    :param forecasts: a numpy array of whole windows forecasts, flattened to a 1D array of shape (n,)
    :param trues: a numpy array of the true values of the series, in shape (n,)
    :param forecast_length: the length of the forecast used for the multi-step forecast
    
    :returns: a dictionary of the MAE and RMSE. The keys are: 'MSE' and 'RMSE'.
    
    """
    errors = {'MAE':[],'RMSE:':[]}
    print('forecasts',forecasts.shape)
    print('trues',trues.shape)
    for i in range(0,forecasts.shape[0]-1,forecast_length):
        errors['MAE'].append(tf.keras.losses.MAE(forecasts[i:i+forecast_length],trues[i:i+forecast_length]).numpy())
        #errors['RMSE'].append(np.sqrt(tf.keras.losses.MSE(forecasts[series_to_predict][i],trues[series_to_predict][i:i+forecast_length]).numpy()))
    
    errors['MAE'] = np.array(errors['MAE']).mean()
    #errors['RMSE'] = np.array(errors['RMSE']).mean()
    
    return errors