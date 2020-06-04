# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:21:14 2020

@author: janak
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:22:45 2020

@author: janak
"""
#Recurrent Neural Network

#Part 1 - Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the training set
dataset_train = pd.read_csv('Accidents_1950_2008.csv')
training_set = dataset_train.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(10, 59):
    X_train.append(training_set_scaled[i-10:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#1 is predictor/indicator here

#Part 2 - Building the RNN

#Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size= 5)

#Part 3 - Making the predictions and visualizing the results

#Getting the real stock price of 2017
dataset_test = pd.read_csv('Accidents_2009_2018.csv')
real_accident_number = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Total_Accidents'], dataset_test['Total_Accidents']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-10:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

#Now prepare variable for prediction
X_test = []
for i in range(10, 20):
    X_test.append(inputs[i-10:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_accident_number = regressor.predict(X_test)
predicted_accident_number = sc.inverse_transform(predicted_accident_number)

#Visualizing the results
plt.plot(real_accident_number, color = 'red', label = 'Real Accident Number')
plt.plot(predicted_accident_number, color = 'blue', label = 'Predicted Accident Number')
plt.title('Yearly Accident Prediction')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.legend()
plt.show()

#Evaluate the model

#Compute the RMSE
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_accident_number, predicted_accident_number))

#Compute Relative Error
mean_predicted_accident_number = predicted_accident_number.mean()
relative_error = rmse/mean_predicted_accident_number





