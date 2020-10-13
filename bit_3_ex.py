import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten


test_path = r'C:\Users\nirro\Desktop\machine learning\topics\topic 35\BitTest2.csv'
test_data_2019=pd.read_csv(test_path)

train_path = r'C:\Users\nirro\Desktop\machine learning\topics\topic 35\BitTrain2.csv'
train_data_2013_2018=pd.read_csv(train_path)
train_2013_2018 = train_data_2013_2018.iloc[:,8].values.reshape(-1, 1)
test_2019 = test_data_2019.iloc[:,8].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range = (0, 1))

train_2013_2018_scaled = scaler.fit_transform(train_2013_2018)
test_2019_scaled = scaler.fit_transform(test_2019)

def fill_nan(data):
    n=len(data)-5
    for ind, i in enumerate(data):
        if ind < 5:
            mask = data[ind:ind+10]
            if np.isnan(i):
                data[ind] = np.nanmean(mask)
        elif ind > n:
            mask=data[ind-10:ind]
            if np.isnan(i):
                data[ind] = np.nanmean(mask)
        elif ind >= 5:
            mask = data[ind-5:ind+5]
            if np.isnan(i):
                data[ind] = np.nanmean(mask)
    return data
new_train = fill_nan(train_2013_2018_scaled)
new_test = fill_nan(test_2019_scaled)

# split train and test to x and y each
def split_x_y(data):
        
    n=60
    list_of_x=[]
    list_of_y=[]
    l = len(data)-n
    for ind in range(l):
        list_of_x.append(data[ind:ind+n])
        list_of_y.append(data[ind+n])
    return np.asarray(list_of_x), np.asarray(list_of_y)
x_train,y_train = split_x_y(new_train)
x_test, y_test = split_x_y(new_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# LSTM model
model = Sequential()
model.add(LSTM(60,input_shape=(x_train.shape[1],1),activation='relu'))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")
model.fit(x_train,y_train,epochs=1,batch_size=32)

# predict the year 2019
predicted_data=model.predict(x_test)
predictions = scaler.inverse_transform(predicted_data)
y_test = scaler.inverse_transform(y_test)


plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')
plt.plot(test_data_2019.iloc[:len(y_test[:200000]),9],y_test[:200000],color="r",label="true result")
plt.plot(test_data_2019.iloc[:len(predictions[:200000]),9],predictions[:200000],color="b",label="predicted result")
# plt.xticks(range(0,y_test.shape[0],10000),rotation=90)
# plt.xticks(np.arange(0,200000,100),rotation=90)
plt.legend()
plt.xlabel("Year 2019")
plt.ylabel("Price Values")
plt.grid(True)
plt.show()
