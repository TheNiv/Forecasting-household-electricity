# -*- coding: utf-8 -*-
"""
Created on Tue May 18 08:53:38 2021

@author: Niv Lifshitz
"""
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import tkinter.font as tkFont
import os
from tkinter import filedialog
import Models
import numpy as np
import DatasetManager as dm
import tensorflow as tf
import pandas as pd
import TimeSeriesHelper as tsh
import DataVisuallizer as dv
from shutil import move
import matplotlib

DATA_PATH = 'household_power_consumption.txt'

global root
root = Tk()
def main():
    print(matplotlib.__version__)
    root.protocol("WM_DELETE_WINDOW", end_program)
    global deafault_font
    deafault_font = tkFont.Font(family="Lucida Grande",size=18)
    
    root.title("Deep Learning Project")
    root.geometry("700x600")
    
    
    title_font = tkFont.Font(family="Lucida Grande",size=24)
    title_label = Label(root,text = "Forecasting Electricity Usage",font = title_font,pady=50)
    title_label.pack()
    
    img = ImageTk.PhotoImage(Image.open('image1.jpg').resize((300,200)))
    panel = Label(image = img)
    panel.pack()
    
    model_selection_btn = Button(root, text = "Choose a model",width = 13,font = deafault_font, command = lambda: open_model_menu_window(root), bg= "#1988ee")
    model_selection_btn.pack(pady=20)
    save_data_btn = Button(root,text="Path to save data",width = 13,font = deafault_font,command=open_save_data_frame,bg = "#22a515")
    save_data_btn.pack(pady=20)

    
    explore_data_btn = Button(root, text = 'Explore data',width = 13, font = deafault_font,command=data_window, bg= "#d81111")
    explore_data_btn.pack(pady=20)
    
    root.mainloop()
    
    
## WINDOWS

def open_save_data_frame():
    window = Toplevel(root)
    
    window.rowconfigure(3,weight=1)
    window.columnconfigure(3,weight=1)
    e = Entry(window,width = 45,font = deafault_font )  
    path = os.path.join(os.getcwd(),'Electricity Data')
    e.insert(END, path)
    
    select_folder_btn = Button(window, text="Choose a folder",font = deafault_font,command = lambda: select_dir(e), bg ="#eb9c1e")
    
    e.grid(row = 1 , column  = 0,pady=20, sticky=E+W)
    select_folder_btn.grid(row=1,column=1,sticky=E+W) 
    
    continue_btn = Button(window,text = "Continue",font = deafault_font,command = lambda:save_data(window,e.get()), bg = "#19e029")
    continue_btn.grid(row=2,column=1, sticky=N+S+E+W, pady=30)

def open_note_frame(path):
    window = Toplevel(root)
    
    window.geometry("600x500")
    
    label = Label(window,text = f"Saves the data file at:\n{path}", font = deafault_font)
    label.pack(pady=50,padx=40)
    
    ok_btn = Button(window,text = "OK" , font = deafault_font,command = lambda: window.destroy() )
    ok_btn.pack(pady=20)
def open_model_menu_window(parent):
    if parent != root:
        parent.destroy()
    else:
        root.withdraw()
    
    window = Toplevel(root)
    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.geometry("600x500")
    
    mulitvariate_btn = Button(window,text = "Multivarite model", height = 4 , command = lambda: actions_menu_window(window,'Multi') , font = deafault_font, bg = "#1988ee")
    mulitvariate_btn.place(bordermode=OUTSIDE ,width = 200, height = 100, x=325, y = 175)
    
    univariate_btn = Button(window, text = "Univariate model", height = 4 ,command = lambda: actions_menu_window(window,'Uni'), font = deafault_font, bg = "#22a515")
    univariate_btn.place(bordermode=OUTSIDE ,width = 200, height = 100, x=75, y = 175)
    
def actions_menu_window(parent, model_type):


    

    parent.destroy()
    window = Toplevel(root)
    window.geometry("600x500")
    window.protocol("WM_DELETE_WINDOW", on_closing)
    
    if model_type == 'Uni':
        train_model_btn = Button(window, text = "Train the model", font = deafault_font, command = lambda: open_train_window(window,'Uni'))
    else:
        train_model_btn = Button(window, text = "Train the model", font = deafault_font, command = lambda: open_train_window(window,'Multi'))
    train_model_btn.place(bordermode=OUTSIDE ,width = 300, height = 100, x=150, y = 25)
    #train_model_btn.pack(pady=20)
    if model_type == 'Multi':
        test_model_btn = Button(window, text = "Test the model", font = deafault_font, command= lambda: select_series_to_predict(window,model_type))
    else:
         test_model_btn = Button(window, text = "Test the model", font = deafault_font, command= lambda: open_test_window(window,model_type))
    test_model_btn.place(bordermode=OUTSIDE ,width = 300, height = 100, x=150, y = 150)
    #test_model_btn.pack(pady=20)
    
    test_multi_forecast_btn = Button(window, text = "Multiforecast with the model", font = deafault_font, command= lambda: select_multi_forecast_options(window,model_type))
    test_multi_forecast_btn.place(bordermode=OUTSIDE ,width = 300, height = 100, x=150, y =275)

def select_series_to_predict(parent,model_type):
    window = Toplevel(parent)
    window.geometry("600x500")
    NAMES = ['Global_active_power','Global_intensity','Voltage']
    variable = StringVar(window)
    variable.set(NAMES[0]) # default value
    
    text = Label(window, text = 'Select with which series do you want to test the model:', font = deafault_font)
    text.pack(pady=25)
    
    w = OptionMenu(window, variable, *NAMES)
    w.config(font = deafault_font)
    menu = window.nametowidget(w.menuname)
    menu.config(font = deafault_font)
    
    w.pack(pady=25)
    if model_type=='Multi':
        btn = Button(window,text = "Test" ,font = deafault_font, command = lambda: open_test_window(window,model_type,variable.get()))
        btn.pack(pady=25)
    elif model_type == 'Uni':
        btn = Button(window,text = "Test" ,font = deafault_font, command = lambda: open_test_window(window,model_type,variable.get()))
        btn.pack(pady=25)
    
    


def select_multi_forecast_options(parent,model_type):
    

    window = Toplevel(parent)
    window.geometry("600x500")
    NAMES = ['Global_active_power','Global_intensity','Voltage']
    text1 = Label(window, text = 'How many  hours to forecast for each window?(max 30)', font  = deafault_font)
    text1.pack(pady=25)
    e = Entry(window,width = 2,font = deafault_font )
    e.insert(END, '8')
    e.pack(pady=30)
    
    if model_type == 'Multi':
        text2 = Label(window, text = 'Select a series to forecast:',font = deafault_font)
        text2.pack(pady=25)
        variable = StringVar(window)
        variable.set(NAMES[0]) # default value
        w = OptionMenu(window, variable, *NAMES)
        w.config(font = deafault_font)
        menu = window.nametowidget(w.menuname)
        menu.config(font = deafault_font)
        w.pack(pady = 30)
    
    
        btn = Button(window, text = 'Forecast' , font  = deafault_font, command = lambda: validate_forecast_len(window,model_type,e.get(),variable.get()))
    else:
        btn = Button(window, text = 'Forecast' , font  = deafault_font, command = lambda: validate_forecast_len(window,model_type,e.get()))
    btn.pack(pady=30)
    


def data_window():
    window = Toplevel(root)
    window.geometry("500x600")

    show_data_btn = Button(window, text = 'The data format', command = display_data_frame, font = deafault_font)
    show_data_btn.pack()
    

#TRAIN, TEST AND MULTI-FORECAST

def open_train_window(parent,model_type, column_to_predict = 'Global_active_power'):
    
    df = pd.read_csv("household_power_consumption.txt", sep=';', low_memory = False, parse_dates= {'date_time' : ['Date', 'Time']},
                                                                                                   infer_datetime_format = True, index_col = 'date_time',
                                                                                                   na_values=['nan','?'])
    dm.fill_missing_values(df,'mean',in_place=True)
    df_resampled = dm.resample_data(df,'H')
    
    if model_type == 'Multi':
        model,fit_parameters,WINDOW_SIZE = Models.multivariate_model()
        df_resampled.drop(columns = ['Sub_metering_1','Sub_metering_2','Sub_metering_3','Global_reactive_power'],inplace = True)
    else:  # model is univariate
        model,fit_parameters,WINDOW_SIZE = Models.univariate_model()
        df_resampled = df_resampled[column_to_predict].to_frame()
    
    df_scaled , scalers = dm.new_scale_data(df_resampled)
    
    train_data,test_data = dm.split_to_train_test(df_scaled,for_train = 0.8,to_numpy = True)
    
    train_X,train_Y  = dm.split_to_windows(train_data ,train_data , window_size = WINDOW_SIZE)
    test_X,test_Y  = dm.split_to_windows(test_data , test_data , window_size = WINDOW_SIZE)
    
    
    
    history = model.fit(train_X,train_Y, **fit_parameters)
    
    forecasts = model.predict(test_X[:-1])
    rescaled_forecasts = dm.rescale_data(forecasts,scalers)
    trues = dm.rescale_data(test_Y[:-1],scalers)
    
    #model.save_weights('GoodWeights')
    mae = tf.keras.losses.MAE(trues[column_to_predict],rescaled_forecasts[column_to_predict])
    rmse = np.sqrt(tf.keras.losses.MSE(trues[column_to_predict],rescaled_forecasts[column_to_predict]))
    mape = tf.keras.losses.MAPE(trues[column_to_predict],rescaled_forecasts[column_to_predict])
    dv.plot(history.history['loss'], title = 'Training loss', xlabel = "Epochs", ylabel = "loss")
    
    loss_info = messagebox.showinfo("info",f"The model has been trained. \n The training loss is:{history.history['loss'][-1]:.3f}")
    
    
    dv.plot(history.history['loss'] , typ = 'line',title = f"Loss over epochs", xlabel = "Epochs" ,
            ylabel = "Loss" )

def open_test_window(parent,model_type,column_to_predict="Global_active_power"):
    

    df = pd.read_csv("household_power_consumption.txt", sep=';', low_memory = False, parse_dates= {'date_time' : ['Date', 'Time']},
                                                                                                   infer_datetime_format = True, index_col = 'date_time',
                                                                                                   na_values=['nan','?'])
    dm.fill_missing_values(df,'mean',in_place=True)
    df_resampled = dm.resample_data(df,'H')
    if model_type == 'Multi':
        model,_,WINDOW_SIZE = Models.multivariate_model()
        model.load_weights('Multivarite_Final_Weights')
        df_resampled.drop(columns = ['Sub_metering_1','Sub_metering_2','Sub_metering_3','Global_reactive_power'],inplace = True)
    else:
        model,_,WINDOW_SIZE = Models.univariate_model()
        model.load_weights('univariate_weights')
        df_resampled = df_resampled[column_to_predict].to_frame()
    
    df_scaled , scalers = dm.new_scale_data(df_resampled)
    
    train_data,test_data = dm.split_to_train_test(df_scaled,for_train = 0.8,to_numpy = True)
    
    train_X,train_Y  = dm.split_to_windows(train_data ,train_data , window_size = WINDOW_SIZE)
    test_X,test_Y  = dm.split_to_windows(test_data , test_data , window_size = WINDOW_SIZE)
    
    forecasts = model.predict(test_X[:-1])
    trues = dm.rescale_data(test_Y[:-1],scalers)
    rescaled_forecasts = dm.rescale_data(forecasts,scalers)
    
    mae = tf.keras.losses.MAE(trues[column_to_predict],rescaled_forecasts[column_to_predict])
    rmse = np.sqrt(tf.keras.losses.MSE(trues[column_to_predict],rescaled_forecasts[column_to_predict]))
    mape = tf.keras.losses.MAPE(trues[column_to_predict],rescaled_forecasts[column_to_predict])
    error_info = messagebox.showinfo("Finished testing",f"MAE:{mae:.3f} \n RMSE:{rmse:.3f} \n MAPE:{mape:.3f}")
    
    
    data = [trues[column_to_predict] , rescaled_forecasts[column_to_predict]]
    legend = ['Real values','Forecasts']
    
    dv.multiple_line_plots(data ,title = f"Hourly {column_to_predict}", xlabel = "Timesteps" ,
            ylabel = f"{column_to_predict}" , legend = legend )

def multi_forecast_window(model_type, forecast_len,column_to_predict):

    df = pd.read_csv("household_power_consumption.txt", sep=';', low_memory = False, parse_dates= {'date_time' : ['Date', 'Time']},
                                                                                                   infer_datetime_format = True, index_col = 'date_time',
                                                                                                   na_values=['nan','?'])
    dm.fill_missing_values(df,'mean',in_place=True)
    df_resampled = dm.resample_data(df,'H')
    if model_type == 'Multi':
        model,_,WINDOW_SIZE = Models.multivariate_model()
        model.load_weights('Multivarite_Final_Weights')
        df_resampled.drop(columns = ['Sub_metering_1','Sub_metering_2','Sub_metering_3','Global_reactive_power'],inplace = True)
    elif model_type == 'Uni':
        model,_,WINDOW_SIZE = Models.univariate_model()
        model.load_weights('UnivariateWeightsFinal')
        df_resampled.drop(columns = ['Global_intensity','Voltage','Sub_metering_1','Sub_metering_2','Sub_metering_3','Global_reactive_power'],inplace = True)
    df_scaled , scalers = dm.new_scale_data(df_resampled)
    
    train_data,test_data = dm.split_to_train_test(df_scaled,for_train = 0.8,to_numpy = True)
    
    train_X,train_Y  = dm.split_to_windows(train_data ,train_data , window_size = WINDOW_SIZE)
    test_X,test_Y  = dm.split_to_windows(test_data , test_data , window_size = WINDOW_SIZE)
    trues = dm.rescale_data(test_Y[:-1],scalers)
    
    
    
    forecasts = tsh.multi_forecast(model,test_X[3000:3500],forecast_length=forecast_len)
    
    
    if model_type == 'Multi':
        forecasts = np.concatenate(forecasts,axis=0)
        forecasts = forecasts.reshape(forecasts.shape[0],forecasts.shape[2])
        forecasts_rescaled_multi = dm.rescale_data(forecasts,scalers)
    else:
        forecasts_rescaled_multi = dm.rescale_data(np.concatenate(forecasts,axis=0),scalers)
        
    errors = tsh.calculate_multi_forecast_errors(forecasts_rescaled_multi[column_to_predict],trues[column_to_predict],forecast_length=forecast_len)

    
    error_info = messagebox.showinfo("Finished forecasting",f" Error results: \n MAE:{errors['MAE']:.3f} \n ")  #RMSE:{errors['RMSE']}
    
    btn_window = Toplevel(root)
    btn_window.geometry("200x200")
    show_random_window_btn = Button(btn_window, command = lambda:dv.plot_random_window(forecasts_rescaled_multi[column_to_predict], trues[column_to_predict],forecast_length = forecast_len),
                                    text = "Show another random window",
                                    font = deafault_font)
    
    show_random_window_btn.pack(pady=30)
def display_data_frame():
     window = Toplevel(root)
     img = ImageTk.PhotoImage(Image.open('data_table.png'))
     panel = Label(window,image = img)
     panel.img = img
     panel.pack()



#OTHER FUNCTIONS
def end_program():
    root.destroy()
    

def select_dir(entry):
    path = filedialog.askdirectory()
    entry.delete(0,END)
    entry.insert(0,path)


    
def save_data(parent,path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    if os.path.exists(path):
        files_in_dir = os.listdir(path)
        if len(files_in_dir) == 0:
            move(os.path.join(os.getcwd(),'household_power_consumption.txt'),os.path.join(path,r'data\household_power_consumption.txt'))
            global DATAPATH
            DATAPATH = os.path.join(path,r'data\household_power_consumption.txt')
            
            open_note_frame(path)
            parent.destroy()
        else:
            open_note_frame(path)
            parent.destroy()
            
    else:
        error_message = Label(parent,text = f"The path {path} does not exist, please enter a valid path",font = deafault_font)
        error_message.grid(row=3 , column=0, sticky=N+S+E+W)
        
    
    
    
    
def validate_forecast_len(window,model_type,string,column_to_predict='Global_active_power'):
    try: 
        forecast_len = int(string)
        multi_forecast_window(model_type,forecast_len,column_to_predict)
    except:
        messagebox.showinfo('error','Please enter an integer for the hours to forecast entry')
        

def select_series_to_train(parent):
    window = Toplevel(parent)
    window.geometry("600x500")
    NAMES = ['Global_active_power','Global_intensity','Voltage']
    text = Label(window, text = 'Select which series do you want to train:', font = deafault_font)
    text.pack(pady=25)
    variable = StringVar(window)
    variable.set(NAMES[0]) # default value
    w = OptionMenu(window, variable, *NAMES)
    w.config(font = deafault_font)
    menu = window.nametowidget(w.menuname)
    menu.config(font = deafault_font)
    w.pack(pady = 30)
    
    btn = Button(window,text = "Train" ,font = deafault_font, command = lambda: open_train_window(window,'Uni',variable.get()))
    btn.pack(pady=30)
    
   
    look_epochs = Label(window, text = "You can see the training progression in the cmd", font = deafault_font )
    look_epochs.pack(pady=20)
    
    

    


    
if __name__ == "__main__":
    main()