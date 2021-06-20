
from tensorflow import keras

def multivariate_model():
    NUM_FEATURES = 3
    compile_parameters = {'loss' : 'mae' , 'optimizer' : keras.optimizers.Adam(), 'metrics' : ['mae'] }
    WINDOW_SIZE = 12
    fit_parameters = {'batch_size' : 256, 'epochs' : 130}
    OPTIMIZER = keras.optimizers.Adam()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64 ,return_sequences=True,input_shape  = (WINDOW_SIZE,NUM_FEATURES)))) 
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(20,activation = "relu"))
    model.add(keras.layers.Dense(NUM_FEATURES))
    
    model.compile(**compile_parameters)
    
    return model, fit_parameters, WINDOW_SIZE


def univariate_model():
    
    compile_parameters = {'loss' : 'mae' , 'optimizer' : keras.optimizers.Adam(), 'metrics' : ['mse'] }
    fit_parameters = {'batch_size' : 256, 'epochs' : 150}
    WINDOW_SIZE = 36
    model = keras.models.Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64 ,return_sequences=True,input_shape  = (WINDOW_SIZE,1)))) 
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(20,activation = "relu"))
    model.add(keras.layers.Dense(1))
    
    model.compile(**compile_parameters)
    
    return model , fit_parameters, WINDOW_SIZE