# -*- coding: utf-8 -*-
"""
Created on Sat May  1 16:00:15 2021

@author: Niv Lifshitz
"""

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import random

def plot_heat_map(df):
    sns.heatmap(df.corr())


def plot(data,typ = 'line',title='',xlabel='',ylabel='',legend = []):
    """
    Plots the data with the type of plot specified.
    
    :param data: a numpy array or python list. The data to plot
    :param typ: a string. The type of plot to use. Should be 'line' or histogram'.
    
    """
    if typ == 'line':
        plt.plot(data)
    elif typ == 'histogram':
        plt.hist(data)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    
    plt.show()
    

def plot_random_window(forecasts,trues,forecast_length=8,title='',xlabel='',ylabel='', legend = []):
    """
    Plots a random window forecasts and true values in one coordinate system.
    
    :param forecasts: a 1D numpy array that contains windows forecasts 
    :param trues: a 1D numpy array that contains the true values of the windows
    :param forecast_length: the forecast length that was used for each window
    """
    random_index = random.randint(0,forecasts.shape[0]//forecast_length -1)
    plt.plot(trues[random_index:random_index+forecast_length])
    plt.plot(forecasts[random_index:random_index+forecast_length])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.show()

def multiple_line_plots(data,title='',xlabel='',ylabel='',legend = []):
    for sequence in data:
        plt.plot(sequence)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.show()
    
def plot_acorr_and_pacf(series,lag = 30):
    """
    Plots acf(autocorrelation) plot and pcf(partial auto correlation) plots for the series for the amount of lag specified
    
    :param series: a numpy array or python list. The series to check its autocorrelation and partial autocorrelation
    :param lag: an integer. The max lag to check
    
    """
    plot_acf(series,lags = 30)
    plot_pacf(series,lags = 30)
    plt.show()