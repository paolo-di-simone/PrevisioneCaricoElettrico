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

        self.other_feature = ["SumOfLines", "DayOfWeek", "Month", "Day", "Hour", "Holiday"]

    
    def check_holiday(self, x):
        if str(x.year) in self.holidays.keys():
            if str(x.month) in self.holidays[str(x.year)].keys():
                if str(x.day) in self.holidays[str(x.year)][str(x.month)]:
                    return 1.0
        return 0.0


    def add_feature(self, df_path):
        df_for_training = pd.DataFrame()

        df = pd.read_csv(df_path, index_col="Time")
        df.index = pd.to_datetime(df.index)

        for line, loads in self.loads_lines.items():
            df_for_training[line] = df[loads].sum(axis=1)

        df_for_training["SumOfLines"] = df_for_training[self.loads_lines.keys()].sum(axis=1)
        df_for_training["DayOfWeek"] = df_for_training.index.to_series().apply(lambda x: x.dayofweek)
        df_for_training["Month"] = df_for_training.index.to_series().apply(lambda x: x.month)
        df_for_training["Day"] = df_for_training.index.to_series().apply(lambda x: x.day)
        df_for_training["Hour"] = df_for_training.index.to_series().apply(lambda x: x.hour)
        df_for_training["Holiday"] = df_for_training.index.to_series().apply(lambda x: self.check_holiday(x))

        return df_for_training
    

    def make_datasets(self):
        df_training = self.add_feature(self.process_df_path)
        df_training.to_csv(self.df_training_path)

        df_plot = self.add_feature(self.unprocess_df_path)
        for line in self.loads_lines.keys():
            df_plot[line] = df_plot[line].replace(0, np.nan)
            df_plot[line] = df_plot[line].clip(lower=0, upper=10000)  
        df_plot.to_csv(self.df_plot_path)


if __name__ == "__main__":
    MakeDataset().make_datasets()
    