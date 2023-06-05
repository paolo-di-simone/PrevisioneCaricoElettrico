import pandas as pd
import os
from datetime import datetime
import calendar
import json

class CheckingData:

    def __init__(self):
        self.data_files_path = os.path.join("Preprocessing", "Dati")
        self.files_shape_path = os.path.join("Preprocessing", "Output", "files_shape.json")
        self.missing_files_path = os.path.join("Preprocessing", "Output", "missing_files.json")

    def missing_days(self, y, m, list_of_days):
        tmp = calendar.monthrange(y, m)[1]
        total_days = set(range(1,tmp+1))
        list_of_days = set(list_of_days)
        return list(total_days.difference(list_of_days))
    
    def check_files(self):
        check_files_shape = {}
        check_missing_files = {}

        for dir_name in os.listdir(self.data_files_path):
            
            dir_files = os.path.join(self.data_files_path, dir_name)
            days = []
            for filename in os.listdir(dir_files):

                print(filename)
            
                plant, f40, year, month, day = filename.split(".")[0].split("_")
                
                days.append(int(day))
                
                file_path = os.path.join(dir_files, filename)
                df_tmp = pd.read_csv(file_path, delimiter="\t")  
                
                if not df_tmp.shape[0] == 5760:
                    check_files_shape[file_path] = df_tmp.shape

            miss_days = self.missing_days(int(year), int(month), days)
            if len(miss_days) > 0:
                check_missing_files[year+" "+month] = miss_days
        

        with open(self.files_shape_path, "w") as outfile:
            json.dump(check_files_shape, outfile, indent=4)
        
        with open(self.missing_files_path, "w") as outfile:
            json.dump(check_missing_files, outfile, indent=4)
