import os
import json
import pandas as pd
import numpy as np

class MakeDataset:

    def __init__(self):
        self.process_df_path = os.path.join("Preprocessing", "Dataset", "dataset_processed.csv")
        self.unprocess_df_path = os.path.join("Preprocessing", "Dataset", "dataset_unprocessed.csv")
        self.df_training_path = os.path.join("Preprocessing", "Dataset", "dataset_training.csv")
        self.df_plot_path = os.path.join("Preprocessing", "Dataset", "dataset_plot.csv")

        with open(os.path.join("Preprocessing", "Input", "loads_lines.json"), "r") as infile:
            self.loads_lines = json.loads(infile.read())

        with open(os.path.join("Preprocessing", "Input", "holidays.json"), "r") as infile:
            self.holidays = json.loads(infile.read())
    
    def check_holiday(self, x):
        if str(x.year) in self.holidays.keys():
            if str(x.month) in self.holidays[str(x.year)].keys():
                if str(x.day) in self.holidays[str(x.year)][str(x.month)]:
                    return 1.0
        return 0.0


    def add_feature(self, df_path):
        df_final = pd.DataFrame()

        df = pd.read_csv(df_path, index_col="Timestamp")
        df.index = pd.to_datetime(df.index)

        #for line, loads in self.loads_lines.items():
        #    df_final[line] = df[loads].sum(axis=1)

        loads = [x for loads in self.loads_lines.values() for x in loads]

        df_final["Carico totale"] = df[loads].sum(axis=1)
        df_final["Temperatura"] = df["Temperatura"]
        df_final["Giorno della settimana"] = df_final.index.to_series().apply(lambda x: x.dayofweek)
        df_final["Mese"] = df_final.index.to_series().apply(lambda x: x.month)
        df_final["Giorno"] = df_final.index.to_series().apply(lambda x: x.day)
        df_final["Ora"] = df_final.index.to_series().apply(lambda x: x.hour)
        df_final["Festivo"] = df_final.index.to_series().apply(lambda x: self.check_holiday(x))

        return df_final
    

    def make_datasets(self):
        df_training = self.add_feature(self.process_df_path)
        df_training.to_csv(self.df_training_path)

        df_plot = self.add_feature(self.unprocess_df_path)
        df_plot["Carico totale"] = df_plot["Carico totale"].clip(lower=0, upper=15000)  
        df_plot["Carico totale"] = df_plot["Carico totale"].replace(0, np.nan)
        df_plot.to_csv(self.df_plot_path)

    