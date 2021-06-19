# -*- coding: utf-8 -*-
"""
Created on Sun May  2 10:57:56 2021

@author: Student
"""
import numpy as np
import tensorflow as tf

def split_sequence(sequences,to_predict, n_steps=1):
    """
    Splits the sequences to windows with the sliding windows and labels(true values).
    
    :param sequences: A numpy array of 1 or more sequences that are used as inputs
    :param to_predict: A numpy array of the sequence/s that are used for the labels(true values)
    :param n_steps: The window size of each window
    
    :returns: A tuple that holds the inputs windows and the corresponding true values of the next timestep
    """
    X, Y = [], []
    for i in range(sequences.shape[0]):
        if i + n_steps >= sequences.shape[0]:
            break
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
    :param scaler: In case of a univariate series, the scaler (sklearn scaler) to use in order to scale the data.
    
    :returns: A 1D numpy array 
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
    
    :returns: a list of numpy arrays of shape: (forecast_length, num features).
    """
    future_forecasts = []
    print('Starts forecasting')
    for i in range(data.shape[0]):
        current_step = data[i]
        future = []
        for j in range(forecast_length):
            forecast = model.predict(current_step.reshape(1,current_step.shape[0],current_step.shape[1]))
            future.append(forecast)
            current_step = np.concatenate([current_step[1:],forecast])
        future_forecasts.append(np.array(future))
        if i % 100 ==0:
            print(f"{i} forecasts completed.")
    return future_forecasts
    

def naive_forecast(series,typ='last_next',window_size = None):
    """
    Uses naive forecast methodes to forecast the series and returnsa numpy array of the forecasts.
    
    :param series: a pandas Series object. The series to forecast.
    :param typ: 'last_next' or 'mean_average'. The type of the forecast to use. 
    :param window_size: an integer. The window_size for the mean average if used.
    :returns: a numpy array with the forecasts.
    
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
    
    :param forecasts: a numpy array of whole windows forecasts, flattened to a 1D array of shape (n,) or 2D array of arrays of shape (n,num_features) so every forecast_length elemtents are a forecasted window
    :param trues: a numpy array of the true values of the series with the same shape as the forecasts
    :param forecast_length: the length of the forecast used for the multi-step forecast
    
    :returns: a dictionary of the MAE . The keys are: 'MAE'
    Note: RMSE was supposed to be calculated and added but I decided not to add it
    
    """
    errors = {'MAE':[],'RMSE:':[]}
    counter = 0 #holds the current window of true values to check
    for i in range(0,forecasts.shape[0]-1,forecast_length):  #Every forecast_length elements are 1 window. Therefore I used a counter.
        errors['MAE'].append(tf.keras.losses.MAE(forecasts[i:i+forecast_length],trues[counter:counter+forecast_length]).numpy())
        counter+=1
        if counter%100 == 0:
            print(f" {counter} windows errors calculated. Current MAE is:{np.array(errors['MAE']).mean()}")
        #errors['RMSE'].append(np.sqrt(tf.keras.losses.MSE(forecasts[series_to_predict][i],trues[series_to_predict][i:i+forecast_length]).numpy()))
    
    errors['MAE'] = np.array(errors['MAE']).mean()
    #errors['RMSE'] = np.array(errors['RMSE']).mean()
    
    return errors