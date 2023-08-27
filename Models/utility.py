
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, OneHotEncoder

import pickle
import os
import json
import datetime

from datetime import datetime, timedelta
from calendar import monthrange


def plot_distribution(df, fields, start_point, end_point, dim, title, xlabel, ylabel, base_path=None):
    
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
	
    if base_path:
        file_name = "".join(title.lower()).replace(" ", "_")
        path = os.path.join(base_path, file_name)
        plt.savefig(path, bbox_inches='tight', transparent=True)
    plt.show()
	

def plot_distribution_load(df, fields, base_path=None):
    
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
        plot_distribution(df, fields, month_start, month_end, dim, title, xlabel, ylabel, base_path)
        
        current_date = current_date.replace(day=1) + timedelta(days=32)
	

def plot_history(h, c1, c2):
    # summarize history for mape
    plt.figure(figsize=(10,5))
    plt.grid()
    plt.plot(h["mape"], color=c1, linewidth="2", marker='o')
    plt.plot(h["val_mape"], color=c2, linewidth="2", marker='o')
    plt.title("Model MAPE")
    plt.ylabel("MAPE (%)")
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


def MAPE(Y_actual, Y_Predicted):
    return np.mean(np.abs((Y_actual - Y_Predicted) / Y_actual)) * 100
	

def import_dataset(dataset_path, date_min, date_max):
	df = pd.read_csv(dataset_path, index_col="Timestamp")
	df.index = pd.to_datetime(df.index)
	date_split_start = pd.to_datetime(date_min)
	date_split_end = pd.to_datetime(date_max)
	return df.loc[(df.index >= date_split_start) & (df.index < date_split_end)]

	
def processing_dataset(df):
	encoder = OneHotEncoder()
	df["Ora Encoded"] = encoder.fit_transform(np.array(df["Ora"]).reshape(-1,1)).toarray().tolist()
	df["Mese Encoded"] = encoder.fit_transform(np.array(df["Mese"]).reshape(-1,1)).toarray().tolist()
	df["Giorno della settimana Encoded"] = encoder.fit_transform(np.array(df["Giorno della settimana"]).reshape(-1,1)).toarray().tolist()
	scaler = RobustScaler()
	df["Linea 1 Scaled"] = scaler.fit_transform(df[["Linea 1"]])
	df["Linea 2 Scaled"] = scaler.fit_transform(df[["Linea 2"]])
	df["Linea 3 Scaled"] = scaler.fit_transform(df[["Linea 3"]])
	df["Carico totale Scaled"] = scaler.fit_transform(df[["Carico totale"]])
	return df, scaler
	
	
def split_dataset(df, date_min, date_max):
	date_split_start = pd.to_datetime(date_min)
	date_split_end = pd.to_datetime(date_max)
	df_train = df.loc[(df.index >= date_split_start) & (df.index < date_split_end)]
	df_validation = df.loc[df.index < date_split_start]
	df_test = df.loc[df.index >= date_split_end]
	print("Train size:", len(df_train), "\nValidation size:", len(df_validation), "\nTest size:", len(df_test))
	return df_train, df_validation, df_test
	
	
def windowed_dataset(df, time_steps):
    y_L1 = []
    y_L2 = []
    y_L3 = []
    y_SumOfLines = []
    x_time_series_L1 = []
    x_time_series_L2 = []
    x_time_series_L3 = []
    x_time_series_SumOfLines = []
    x_other_feature = []
    for i in range(len(df)-time_steps):
        
        x_time_series_L1.append(df["Linea 1Scaled"].iloc[i:i+time_steps])
        x_time_series_L2.append(df["Linea 2 Scaled"].iloc[i:i+time_steps])
        x_time_series_L3.append(df["Linea 3 Scaled"].iloc[i:i+time_steps])
        x_time_series_SumOfLines.append(df["Carico totale Scaled"].iloc[i:i+time_steps])
        
        tmp = []
        tmp.append(df["Festivo"].iloc[i+time_steps])
        tmp.extend(df["Ora Encoded"].iloc[i+time_steps])
        tmp.extend(df["Giorno della settimana Encoded"].iloc[i+time_steps])
        tmp.extend(df["Mese Encoded"].iloc[i+time_steps])
        x_other_feature.append(tmp)
        
        y_L1.append(df["Linea 1 Scaled"].iloc[i+time_steps])
        y_L2.append(df["Linea 2 Scaled"].iloc[i+time_steps])
        y_L3.append(df["Linea 3 Scaled"].iloc[i+time_steps])
        y_SumOfLines.append(df["Carico totale Scaled"].iloc[i+time_steps])
    
    return {
        "xL1": np.array(x_time_series_L1),
        "xL2": np.array(x_time_series_L2),
        "xL3": np.array(x_time_series_L3),
        "xCaricoTotale": np.array(x_time_series_SumOfLines),
        "FeatureAggiuntive": np.array(x_other_feature),
        "yL1": np.array(y_L1),
        "yL2": np.array(y_L2),
        "yL3": np.array(y_L3),
        "yCaricoTotale": np.array(y_SumOfLines)
    }
	

def get_x(data, line):
    return [data["x"+line], data["FeatureAggiuntive"]]
	

def get_y(data, line):
    return data["y"+line]
















