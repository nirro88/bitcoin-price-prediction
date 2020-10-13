import numpy as np
import pandas as pd
from datetime import datetime
def take_small_data():
    file_path = r'C:\Users\nirro\Desktop\machine learning\topics\topic 35\bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    all_data=pd.read_csv(file_path)
    all_data['date'] = pd.to_datetime(all_data['Timestamp'],unit='s').dt.date
    d=all_data.loc[0,'date']
    list_of_year=[]
    for i in range(len(all_data)):
        d=all_data.date[i]
        list_of_year.append(d.year)
    all_data['year']=list_of_year
    all_data_2=all_data[all_data.year>2017]
    train=all_data_2[all_data_2.year<2019]
    test=all_data_2[all_data_2.year == 2019]
    train.to_csv(path_or_buf=r'C:\Users\nirro\Desktop\machine learning\topics\topic 35\BitTrain2.csv')
    test.to_csv(path_or_buf=r'C:\Users\nirro\Desktop\machine learning\topics\topic 35\BitTest2.csv')

take_small_data()