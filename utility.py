
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler, OneHotEncoder

import pickle
import os
import json
import datetime

from datetime import datetime, timedelta
from calendar import monthrange
import matplotlib.dates as mdates
import locale

locale.setlocale(locale.LC_TIME, 'it_IT')

dict_dow = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]

counter = 0
def make_path(base_path, file_name):
    global counter
    counter += 1
    file_name = str(counter) + "-" + file_name  
    return os.path.join(base_path, file_name)
	

def get_df_errors(df_prediction, field_true, field_prediction):
    start_date = df_prediction.index.min()
    end_date = df_prediction.index.max()

    data_error = []

    current_date = start_date
    while current_date <= end_date:
        day_start = current_date.replace(hour=0)
        day_end = current_date.replace(hour=23)

        df_day = df_prediction.loc[day_start:day_end]

        mape = MAPE(df_day[field_true], df_day[field_prediction])
        mae = MAE(df_day[field_true], df_day[field_prediction])

        data_error.append({
            "date": current_date,
            "mape": mape,
            "mae": mae
        })

        current_date += timedelta(days=1)

    df_error = pd.DataFrame(data_error)
    df_error = df_error.set_index("date")
    df_error.index = pd.to_datetime(df_error.index)
    df_error["DoW"] = df_error.index.to_series().apply(lambda x: dict_dow[x.dayofweek])
    return df_error


def get_df_error_days_of_week(df_prediction, field_true, field_prediction):
    df_data = [] 
    
    for i in range(0,7):
        df_day_of_week = df_prediction[df_prediction.index.dayofweek == i]
        mape = MAPE(df_day_of_week[field_true], df_day_of_week[field_prediction])
        mae = MAE(df_day_of_week[field_true], df_day_of_week[field_prediction])
        df_data.append({
            "DoW": dict_dow[i],
            "mape": mape,
            "mae": mae
        })

    return pd.DataFrame(df_data)


def bar_plot(keys, values, ylabel, title, color, dim, linewidth, base_path=None):
    
    plt.figure(figsize=dim) 
    plt.ylabel(ylabel)
    plt.title(title)
    
    p = plt.bar(keys, values, width= 0.5, color=color, edgecolor="#000000", linewidth=linewidth, align='center')
    
    plt.bar_label(p, label_type='edge', padding=5)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.margins(y=0.1)
    plt.tick_params(left=True, labelleft=True)

    if base_path:
        if title:
            file_name = "".join(title.lower()).replace(" ", "_")
            file_name = file_name.replace(":", "_")
        else:
            file_name = "plot"
        path = make_path(base_path, file_name)
        plt.savefig(path, bbox_inches='tight', transparent=True)

    plt.show()


def multiple_bar_plot(keys, values, labels, bar_width, xlabel, ylabel, title, colors, dim, base_path=None):
    
    plt.figure(figsize=dim) 
    
    x = np.arange(len(keys))    
    for i, label in enumerate(labels):
        p = plt.bar(x + i * bar_width, values[i], width=bar_width, color=colors[i], edgecolor="#000000", linewidth=1, align="center", label=label)
        plt.bar_label(p, label_type="edge", padding=5) 

    plt.margins(y=0.1)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(x + (bar_width * (len(labels) - 1)) / 2, keys)

    plt.legend()

    if base_path:
        if title:
            file_name = "".join(title.lower()).replace(" ", "_")
            file_name = file_name.replace(":", "_")
        else:
            file_name = "plot"
        path = make_path(base_path, file_name)
        plt.savefig(path, bbox_inches='tight', transparent=True)

    plt.show()


def box_plot_error(df_error, field, title, ylabel, dim, base_path=None):

    plt.figure(figsize=dim) 

    plt.title(title)
    plt.xlabel("Giorno della settimana")
    plt.ylabel(ylabel)

    data_for_plot = [df_error[df_error.index.weekday == i][field] for i in range(0,7)]
    sns.boxplot(data=data_for_plot, palette="light:g")

    plt.xticks([0,1,2,3,4,5,6], ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"])

    if base_path:
        if title:
            file_name = "".join(title.lower()).replace(" ", "_")
            file_name = file_name.replace(":", "_")
        else:
            file_name = "plot"
        path = make_path(base_path, file_name)
        plt.savefig(path, bbox_inches='tight', transparent=True)

    plt.show()


def plot_multiple_distribution(df, fields, start_point, end_point, dim, title, xlabel, ylabel, colors, markers, markersize, linewidth, base_path=None):
        
    df_tmp = df.loc[start_point:end_point]
    
    plt.figure(figsize=dim) 
    plt.grid()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
	
    for field, color, marker in zip(fields, colors, markers):
        plt.plot(df_tmp.index, df_tmp[field], marker, markersize=markersize, linewidth=linewidth, label=field, color=color)
            
    if len(df_tmp) < 25:
        plt.xticks(df_tmp.index, df_tmp.index.strftime('%H:%M'), rotation=45)
    else:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(df_tmp.index, rotation=45)  
            
    plt.legend()

    if base_path:
        if title:
            file_name = "".join(title.lower()).replace(" ", "_")
            file_name = file_name.replace(":", "_")
        else:
            file_name = "plot"
        path = make_path(base_path, file_name)
        plt.savefig(path, bbox_inches='tight', transparent=True)
		
    plt.show()


def plot_distribution(df, fields, start_point, end_point, dim, title, xlabel, ylabel, base_path=None, marker=False, colors=None, linewidth=2, grid=True):
        
    df_tmp = df.loc[start_point:end_point]
    
    plt.figure(figsize=dim) 
    if grid:
        plt.grid()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
	
    if colors != None:
        for field, color in zip(fields, colors):
            if marker:
                plt.plot(df_tmp.index, df_tmp[field], marker='o', markersize="5", linewidth=linewidth, label=field, color=color)
            else:
                plt.plot(df_tmp.index, df_tmp[field], linewidth=linewidth, label=field, color=color)
    else:
        for field in fields:
            if marker:
                plt.plot(df_tmp.index, df_tmp[field], marker='o', markersize="5", linewidth=linewidth, label=field)
            else:
                plt.plot(df_tmp.index, df_tmp[field], linewidth=linewidth, label=field)
            
    if marker and len(df_tmp) < 25:
        plt.xticks(df_tmp.index, df_tmp.index.strftime('%H:%M'), rotation=45)
    else:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=45)  
            
    if len(fields) > 1:
        plt.legend()

    if base_path:
        if title:
            file_name = "".join(title.lower()).replace(" ", "_")
            file_name = file_name.replace(":", "_")
        else:
            file_name = "plot"
        path = make_path(base_path, file_name)
        plt.savefig(path, bbox_inches='tight', transparent=True)
		
    plt.show()
	

def plot_distribution_load(df, fields, path=None, marker=False, colors=None):
    
    start_date = df.index.min()
    end_date = df.index.max()
    
    current_date = start_date
    while current_date <= end_date:
        month_start = current_date.replace(day=1, hour=0)
        month_end = current_date.replace(day=monthrange(current_date.year, current_date.month)[1], hour=23)
        
        title = "Curva di carico " + month_start.strftime('%B') + " " + str(month_start.year)
        xlabel = "Giorno"
        ylabel = "W"
        dim = (20,7)
        plot_distribution(df, fields, month_start, month_end, dim, title, xlabel, ylabel, path, marker, colors)
        
        current_date = current_date.replace(day=1) + timedelta(days=32)


def plot_history(h, c1, c2, dim, base_path=None):

    plt.figure(figsize=dim)
    plt.grid()
    plt.plot(np.array(h["mape"])-10, color=c1, linewidth="3", marker='o', markersize="5")
    plt.plot(np.array(h["val_mape"])-10, color=c2, linewidth="3", marker='o', markersize="5")
    plt.ylabel("MAPE (%)")
    plt.xlabel("Epoca")
    plt.legend(["Training set", "Validation set"], loc="upper right")
	
    if base_path:
        path = make_path(base_path, "mape")
        plt.savefig(path, bbox_inches='tight', transparent=True)
	
    plt.show()

    plt.figure(figsize=dim)
    plt.grid()
    plt.plot(h["loss"], color=c1, linewidth="3", marker='o', markersize="5")
    plt.plot(h["val_loss"], color=c2, linewidth="3", marker='o', markersize="5")
    plt.ylabel("MSE (loss)")
    plt.xlabel("Epoca")
    plt.legend(["Training set", "Validation set"], loc="upper right")
	
    if base_path:
        path = make_path(base_path, "loss")
        plt.savefig(path, bbox_inches='tight', transparent=True)
	
    plt.show()


def make_tuning_csv(directory, project_name, file_name):
    df_data = []
	
    i = 0
    path = os.path.join(directory, project_name)
    for trial in os.listdir(path):
        json_string = {}
        if "trial" in trial:
            with open(os.path.join(path, trial, "trial.json"), "r") as file:
                data = json.load(file)

                t = data["hyperparameters"]["values"]["regularizer_type"]
                if data["hyperparameters"]["values"]["regularizer_type"] == "none":
                    t = None
				
                json_string["configuration"] = "T"+str(i)
                json_string["days"] = data["hyperparameters"]["values"]["days"]
                json_string["batch_size"] = data["hyperparameters"]["values"]["batch_size"]
                json_string["learning_rate"] = data["hyperparameters"]["values"]["learning_rate"]
                json_string["dropout_rate"] = round(data["hyperparameters"]["values"]["dropout_rate"], 2)
                json_string["regularizer_type"] = t
                json_string["regularizer_param"] = data["hyperparameters"]["values"].get("regularizer_param")
                json_string["BEST STEP"] = data["best_step"]
                json_string["MAPE"] = round(data["score"], 2)

                df_data.append(json_string)
				
                i+=1

    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(path, file_name), index=False)


def print_df_tuning(df):
    i = 0
    for index, row in df.iterrows():
        print("\hline")
        
        print(
            "T"+str(i), "&",
            row["days"], "&",
            row["batch_size"], "&",
            row["learning_rate"], "&",
            row["dropout_rate"], "&",
            row["regularizer_type"], "&",
            row["regularizer_param"], "&",
            round(row["MAPE"], 2), "\\\\"
        )
        
        i += 1
        
    print("\hline")
	

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
	

def MAE(Y_actual, Y_Predicted):
    return np.mean(np.abs(Y_Predicted - Y_actual))


def RMSE(Y_actual, Y_Predicted):
    return np.sqrt(np.mean((Y_Predicted - Y_actual) ** 2))
	

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
	#df["Linea 1 Scaled"] = scaler.fit_transform(df[["Linea 1"]])
	#df["Linea 2 Scaled"] = scaler.fit_transform(df[["Linea 2"]])
	#df["Linea 3 Scaled"] = scaler.fit_transform(df[["Linea 3"]])
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
    #y_L1 = []
    #y_L2 = []
    #y_L3 = []
    y_SumOfLines = []
    #x_time_series_L1 = []
    #x_time_series_L2 = []
    #x_time_series_L3 = []
    x_time_series_SumOfLines = []
    x_other_feature = []
    for i in range(len(df)-time_steps):
        
        #x_time_series_L1.append(df["Linea 1Scaled"].iloc[i:i+time_steps])
        #x_time_series_L2.append(df["Linea 2 Scaled"].iloc[i:i+time_steps])
        #x_time_series_L3.append(df["Linea 3 Scaled"].iloc[i:i+time_steps])
        x_time_series_SumOfLines.append(df["Carico totale Scaled"].iloc[i:i+time_steps])
        
        tmp = []
        tmp.append(df["Festivo"].iloc[i+time_steps])
        tmp.extend(df["Ora Encoded"].iloc[i+time_steps])
        tmp.extend(df["Giorno della settimana Encoded"].iloc[i+time_steps])
        tmp.extend(df["Mese Encoded"].iloc[i+time_steps])
        x_other_feature.append(tmp)
        
        #y_L1.append(df["Linea 1 Scaled"].iloc[i+time_steps])
        #y_L2.append(df["Linea 2 Scaled"].iloc[i+time_steps])
        #y_L3.append(df["Linea 3 Scaled"].iloc[i+time_steps])
        y_SumOfLines.append(df["Carico totale Scaled"].iloc[i+time_steps])
    
    return {
        #"xL1": np.array(x_time_series_L1),
        #"xL2": np.array(x_time_series_L2),
        #"xL3": np.array(x_time_series_L3),
        "xCaricoTotale": np.array(x_time_series_SumOfLines),
        "FeatureAggiuntive": np.array(x_other_feature),
        #"yL1": np.array(y_L1),
        #"yL2": np.array(y_L2),
        #"yL3": np.array(y_L3),
        "yCaricoTotale": np.array(y_SumOfLines)
    }
	

def get_x(data, line):
    return [data["x"+line], data["FeatureAggiuntive"]]
	

def get_y(data, line):
    return data["y"+line]
















