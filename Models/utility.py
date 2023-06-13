
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder

import pickle
import os
import json
import datetime

from datetime import datetime, timedelta
from calendar import monthrange


def save_scaler(scaler, path):
    with open(path, "wb") as file:
        pickle.dump((scaler.data_min_, scaler.data_max_), file)
		

def read_scaler(path):
    with open(path, "rb") as file:
        data_min, data_max = pickle.load(file)
        scaler = MinMaxScaler()
        scaler.data_min_ = data_min
        scaler.data_max_ = data_max
        return scaler


def plot_distribution(df, fields, start_point, end_point, dim, title, xlabel, ylabel, path):
    
    df_tmp = df.loc[start_point:end_point]
    
    plt.figure(figsize=dim) 
    plt.grid()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    for field in fields:
        plt.plot(df_tmp.index, df_tmp[field], linewidth="2", label=field)
    
    plt.xticks(rotation=45)
    
    plt.legend()
    
    #file_name = "".join(title.lower()).replace(" ", "_")
    #path = make_path(plot_path, file_name)
    #plt.savefig(path, bbox_inches='tight', transparent=True)
    plt.show()
	

def plot_distribution_load(df, fields, path):
    
    start_date = df.index.min()
    end_date = df.index.max()
    
    current_date = start_date
    while current_date <= end_date:
        month_start = current_date.replace(day=1, hour=0)
        month_end = current_date.replace(day=monthrange(current_date.year, current_date.month)[1], hour=23)
        
        title = "Curva di carico " + month_start.strftime('%B') + " " + str(month_start.year)
        xlabel = "Tempo"
        ylabel = "W"
        dim = (30,10)
        plot_distribution(df, fields, month_start, month_end, dim, title, xlabel, ylabel, path)
        
        current_date = current_date.replace(day=1) + timedelta(days=32)
	

def plot_history(h, c1, c2):
    # summarize history for mape
    plt.figure(figsize=(10,5))
    plt.grid()
    plt.plot(h["mae"], color=c1, linewidth="2", marker='o')
    plt.plot(h["val_mae"], color=c2, linewidth="2", marker='o')
    plt.title("Model MAE")
    plt.ylabel("MAE")
    plt.xlabel("epochs")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(10,5))
    plt.grid()
    plt.plot(h["loss"], color=c1, linewidth="2", marker='o')
    plt.plot(h["val_loss"], color=c2, linewidth="2", marker='o')
    plt.title("Model Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()
	

def save_history(history, path):
    with open(path, "w") as outfile:
        json.dump(history, outfile, indent=4)
		

def read_history(path):
    with open(path, "r") as infile:
        return json.loads(infile.read())


def add_prediction(df, prediction, time_steps, label):
    df = df.iloc[time_steps:]
    df[label] = prediction
    return df

















