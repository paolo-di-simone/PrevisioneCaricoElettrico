import pandas as pd
import os
from datetime import datetime
import numpy as np
import json

class Preprocessing:

    def __init__(self):
        self.data_files_path = os.path.join("Preprocessing", "Dati")
        self.process_df_path = os.path.join("Preprocessing", "Dataset", "dataset_processed.csv")
        self.unprocess_df_path = os.path.join("Preprocessing", "Dataset", "dataset_unprocessed.csv")

        self.best_file_path = os.path.join("Preprocessing", "Output", "best_file.txt")
        self.deleted_days = os.path.join("Preprocessing", "Output", "deleted_days.json")

        with open(os.path.join("Preprocessing", "Input", "columns_translation.json"), "r") as infile:
            self.columns_translation = json.loads(infile.read())
        
        with open(os.path.join("Preprocessing", "Input", "list_of_loads.json"), "r") as infile:
            self.list_of_loads = json.loads(infile.read())

        self.other_feature = ["Temperatura"]

        with open(os.path.join("Preprocessing", "Input", "min_power.json"), "r") as infile:
            self.min_power = json.loads(infile.read())

        with open(os.path.join("Preprocessing", "Input", "max_power.json"), "r") as infile:
            self.max_power = json.loads(infile.read())

        self.df = pd.DataFrame()
        self.best_days = [("",np.inf),("",np.inf),("",np.inf),("",np.inf),("",np.inf),("",np.inf),("",np.inf)]
        self.df_avg_hour = pd.DataFrame()


    def make_timestamp(self, field, year, month, day):
        time, ampm = field.split(" ")
        h, m, s = time.split(":")
        if h == "12":
            h = "00"
        if ampm == "PM":
            h = str(int(h)+12)
        dtime = datetime(int(year), int(month), int(day), int(h), int(m), int(s))
        return str(dtime)
    

    def distance(self, d1, d2):
        sum_of_squares = 0
        for field in self.list_of_loads:
            diff = d1[field] - d2[field]
            sum_of_squares += diff ** 2
        return np.sqrt(sum_of_squares)
    

    def read_data(self):
        list_of_df = []
        for dir_name in os.listdir(self.data_files_path):
            
            dir_files = os.path.join(self.data_files_path, dir_name)
            for filename in os.listdir(dir_files):

                print(filename)
            
                plant, f40, year, month, day = filename.split(".")[0].split("_")
                
                file_path = os.path.join(dir_files, filename)
                df_tmp = pd.read_csv(file_path, delimiter="\t")  

                df_tmp["Timestamp"] = df_tmp["Time"].apply(lambda x: self.make_timestamp(x, year, month, day))
                df_tmp = df_tmp.set_index("Timestamp")
                
                df_tmp = df_tmp.rename(columns=self.columns_translation)

                df_tmp = df_tmp.loc[:, self.list_of_loads + self.other_feature]

                list_of_df.append(df_tmp)

        self.df = pd.concat(list_of_df)
        self.df.index = pd.to_datetime(self.df.index)

        self.df.to_csv("prova.csv")


    def delete_outlier(self):
        for load in self.list_of_loads:
            self.df[load] = self.df[load].where(
                (self.df[load] >= self.min_power[load]) & (self.df[load] <= self.max_power[load]), np.nan
            )


    def search_best_files(self):
        df_mean_day_of_week = self.df.groupby(self.df.index.dayofweek).mean()
        df_mean_day = self.df.resample("D").mean()
        for index, row in df_mean_day.iterrows():
            dist = self.distance(row, df_mean_day_of_week.loc[index.dayofweek])
            if dist < self.best_days[index.dayofweek][1]:
                self.best_days[index.dayofweek] = (str(index.year)+"-"+str(index.month)+"-"+str(index.day), dist)
        
        with open(self.best_file_path, "w") as outfile:
            for item in self.best_days:
                outfile.write("%s\n" % str(item))
    

    def fill_missing_days(self):
        start_date = self.df.index.min().date()
        end_date = self.df.index.max().date()
        complete_index = pd.date_range(start=start_date, end=end_date, freq="D")
        missing_dates = complete_index.difference(self.df.index.date)

        list_of_df_missing_date = []
        for miss_date in missing_dates:
            df_tmp = self.df.loc[self.best_days[miss_date.weekday()][0]]
            df_tmp.index = df_tmp.index.to_series().apply(
                lambda x: x.replace(year=miss_date.year, month=miss_date.month, day=miss_date.day)
            )
            list_of_df_missing_date.append(df_tmp)
        
        self.df = pd.concat([self.df] + list_of_df_missing_date)
        self.df.sort_index(inplace=True)


    def fill_empty_value(self, df_main, df_tmp):
        for index, row in df_main.iterrows():
            for load in self.list_of_loads:
                if pd.isna(row[load]):
                    df_main.loc[index][load] = df_tmp.loc[index.hour][load]
        return df_main


    def fill_deleted_outlier(self):
        # Creazione dataset mediato orario
        df_avg = self.df.resample("H").mean()

        # Split dataframe in weekday-weekend
        df_weekday_avg = df_avg[df_avg.index.weekday < 5]
        df_weekend_avg = df_avg[df_avg.index.weekday >= 5]
        df_tmp_weekday = self.df[self.df.index.weekday < 5]
        df_tmp_weekend = self.df[self.df.index.weekday >= 5]

        # Media carichi per ogni orario
        df_weekday_avg_hour = df_tmp_weekday.groupby(df_tmp_weekday.index.hour).mean()
        df_weekend_avg_hour = df_tmp_weekend.groupby(df_tmp_weekend.index.hour).mean()

        # Sostituzione valori np.nan con valori medi
        df_weekday_avg = self.fill_empty_value(df_weekday_avg, df_weekday_avg_hour)
        df_weekend_avg = self.fill_empty_value(df_weekend_avg, df_weekend_avg_hour)
                
        # Costruzione dataset mediato orario
        self.df_avg_hour = pd.concat([df_weekday_avg, df_weekend_avg])
        self.df_avg_hour.sort_index(inplace=True)


    def make_process_dataset(self):
        # Lettura file di dati e creazione dataframe
        print("[0] Lettura file di dati e creazione dataframe")
        self.read_data()

        # Eliminazione outlier
        print("[1] Eliminazione outlier")
        self.delete_outlier()

        # Ricerca dei giorni più rappresentativi (che si avvicinano di più alla media)
        print("[2] Ricerca dei giorni più rappresentativi (che si avvicinano di più alla media)")
        self.search_best_files()

        # Riempimento giornate mancanti con i giorni più rappresentativi
        print("[3] Riempimento giornate mancanti con i giorni più rappresentativi")
        self.fill_missing_days()
                
        # Riempimento outlier eliminati e costruzione dataset mediato orario
        print("[4] Riempimento outlier eliminati e costruzione dataset mediato orario")
        self.fill_deleted_outlier()

        # Salvataggio dataframe
        print("[5] Salvataggio dataframe in " + str(self.process_df_path))
        self.df_avg_hour.to_csv(self.process_df_path)
       
    
    def make_not_process_dataset(self):
        # Lettura file di dati e creazione dataframe
        print("[0] Lettura file di dati e creazione dataframe")
        self.read_data()

        # Creazione dataset mediato
        print("[1] Creazione dataset mediato")
        df_avg = self.df.resample("H").mean()

        # Salvataggio dataframe
        print("[2] Salvataggio dataframe in " + str(self.unprocess_df_path))
        df_avg.to_csv(self.unprocess_df_path)
