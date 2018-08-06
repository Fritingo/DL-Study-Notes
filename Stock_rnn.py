# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
print(training_set_scaled)
print(np.shape(training_set_scaled))

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
#set timesteps(60)
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])#to 2d
    y_train.append(training_set_scaled[i, 0])
print(X_train)
print(np.shape(X_train))
print(y_train)
print(np.shape(y_train))
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping to 3d
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train)
print(np.shape(X_train))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
#layer(LSTM):模型.add(LSTM(hidden_layer_neurons數,返回序列,輸入參數數))
#return_sequences(False 返回單個 hidden state)(True 返回全部 hidden state)
#hidden state (目前NN輸出參數、下一個時間NN考慮參數)cell state(影響閘、hidden state的參數)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))#X_train.shape[1] is timesteps
regressor.add(Dropout(0.2))#捨棄神經元20%(避免overfit)

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
#layer(全連接):模型.add(Dense(hidden_layer_neurons數)
regressor.add(Dense(units = 1))

# Compiling the RNN
#模型.compile(優化器,損失函數)均方誤差(mean_squared_error)預測值與實際值的差距之平均值
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
#模型.compile(X_train, y_train, 批量, 代)
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#Save model
regressor.save("Stock_rnn.h5")

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
#將讀取的資料直向合併
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()#標示線名稱
plt.show()
