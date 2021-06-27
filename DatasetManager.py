
import pandas as pd
import sklearn.preprocessing as preprocessing
import numpy as np

class Error(Exception):
    """Base class for exceptions in this module."""
    pass


def resample_data(df,rule='H'):
    """
    Returns a resampled dataframe or series by the rule specified.
	:param pandas DataFrame df: the dataframe to resample
	:param str rule: the new frequency of the series. can be DateOffset, Timedelta or a string.
    
    :returns: a DataFrame that is resampled for the mean value for the frequency that is specified
    
    For more info about resampling options check the pandas documention at DateOffset objects:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    
    """
    return df.resample(rule).mean()

def get_sum_of_missing_values(df):
    """
    Returns the number of missing values in the dataframe
    
	:param pandas DataFrame df: the dataframe you want to check for missing values.
    
    :returns: an integer - the sum of missing values in the DataFrame
	"""
    return df.isnull().sum().sum()



def fill_missing_values(df,method='mean',in_place = False):
    """
    Fills the missing value in the DataFrame with the method specified. Method is mean by default.
    
	:param pandas DataFrame df: the dataframe to fill the missing values in
	:param str method ("mean"/"last"): the method to use to fill the missing values
    """
    if method == 'mean':
        return df.fillna(df.mean(),inplace = in_place)
    elif method == 'last':
         return df.fillna(method='ffill',inplace=in_place)
    elif method == 'next':
        return df.fillna(method='bfill',inplace=in_place)
    else:
        raise Exception('Method is not valid')


def new_scale_data(df,method='MinMax',in_place = False):
    """
    Scales the data in the DataFrame with the method specified.
    
	:param pandas DataFrame df - the dataframe to scale
	:param str method("MinMax"/"Standardize"/"Normallize") -  the scaling method to use
	
    :returns: a tuple of a DataFrame with the scaled values and  a dictionary of the series names and their corresponding scalers
    """
    scalers = {}
    new_df = pd.DataFrame()
    if(in_place):
        new_df = df
    if(method == 'MinMax'):
        scaler = preprocessing.MinMaxScaler()
    elif method == 'Standardize':
        scaler = preprocessing.StandardScaler()
    elif method == 'Normallize':
        scaler = preprocessing.Normalizer()
    else:
        raise Exception('Method is not valid')
    
        
    for column in df.columns:
       scaler = preprocessing.MinMaxScaler()
       scaler = scaler.fit(df[column].to_numpy().reshape(-1, 1))
       new_df[column] = scaler.transform(df[column].to_numpy().reshape(-1, 1)).reshape(df.shape[0],)
       scalers[column] = scaler
    if(in_place):
        return scalers
    return new_df,scalers


def rescale_data(data,scalers):
    """
    Rescales the data to its original range of values.
    
	:param numpy array data: the data to rescale.
    :param dictionary scalers: a dictionary of the series names and their corresponding scalers
	 
    :returns: returns a dictionary of the series names and their corresponding rescaled series
    """
    true_values = {}
    num_features = len(scalers.keys())
    for k,i in zip(scalers,range(num_features)):
        true_values[k] = scalers[k].inverse_transform(data[:,i].reshape(-1,1)).reshape((data.shape[0],))
    return true_values
        

def split_to_train_test(df,for_train = 0.8,to_numpy = False):
    """
    Returns a tuple of 2 DataFrames or numpy arrays, for_train is the part of the data that will be in the train DataFrame or numpy array.
    
	:param pandas DataFrame df: the dataframe to split
	:param float for_train: the part of the data that will be used for the train data, can be a number between 0 and 1
	:param bool to_numpy: whether to return the data as numpy arrays. default is False
    
    :returns: a tuple of DataFrames or numpy arrays that holds the data that should be used for training and testing
    """
    if for_train < 0 or for_train > 1:
        raise Exception('for_train parameter must be in range 0-1')
    split_index = int(for_train*df.shape[0])
	
    if to_numpy:
        return df.iloc[:split_index].values, df.iloc[split_index:].values
    return df.iloc[:split_index], df.iloc[split_index:]
    

def split_to_windows(sequences,to_predict, window_size=1):
    """
    
    Creates an array of windows from the dataset and an array of outputs using the sliding window technique.
    
    if sequences is 2D numpy array:
    X shape in the end of the function: (sequences.shape[0]-window_size, window_size, sequences.shape[1])
    
    if sequences is 1D numpy array:
    X shape in the end of the function: (sequences.shape[0]-window_size, window_size)
    
    :param numpy array sequences: the series' to split to windows
    :param numpy array to_predict: the series of the desired output for each window
    :param int window_size: the size of each window
    
    :returns: an array that consists  of (the number of timesteps in the data - window_size) windows of length window_size
    """
    X, Y = [], []
    
    for i in range(sequences.shape[0]):
        if(i + window_size) >= sequences.shape[0]:
            break
          # Divide sequence between data (input) and labels (output)
        seq_X, seq_Y = sequences[i: i + window_size],to_predict[i + window_size]
                     
        X.append(seq_X)
        Y.append(seq_Y)
    return np.array(X), np.array(Y)
