import numpy as np
import pandas as pd
from datetime import datetime

file_path = r'C:\Users\nirro\Desktop\machine learning\topics\topic 35\bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
all_data=pd.read_csv(file_path)
all_data['date'] = pd.to_datetime(all_data['Timestamp'],unit='s').dt.date
d=all_data.loc[0,'date']
list_of_year=[]
for i in range(len(all_data)):
    d=all_data.date[i]
    list_of_year.append(d.year)
all_data['year']=list_of_year
all_data_2=all_data[all_data.year>2013]
all_data_3=all_data_2[all_data_2.year<2015]
all_data_3.to_csv(file_name,path_or_buf=r'C:\Users\nirro\Desktop\machine learning\topics\topic 35',compression='gzip')