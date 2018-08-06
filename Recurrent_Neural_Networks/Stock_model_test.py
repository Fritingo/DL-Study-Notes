import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
regressor = load_model("Stock_rnn.h5")
# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
# print(training_set)
# print(np.shape(training_set))

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
#轉乘 0 ~ 1 之間(X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))X_scaled = X_std * (max - min) + min)
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# print(training_set_scaled)
# print(np.shape(training_set_scaled))

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
#set timesteps(60)
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])#to 2d
    y_train.append(training_set_scaled[i, 0])
# print(X_train)
# print(np.shape(X_train))
print(y_train)
print(np.shape(y_train))
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
# print(real_stoc_price)
# print(np.shape(real_stock_price))

# Getting the predicted stock price of 2017
#將讀取的資料直向合併
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# print(dataset_total)
# print(np.shape(dataset_total))
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
print(inputs)
print(np.shape(inputs))
inputs = inputs.reshape(-1,1)
print(inputs)
print(np.shape(inputs))
inputs = sc.transform(inputs)
print(inputs)
print(np.shape(inputs))
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
