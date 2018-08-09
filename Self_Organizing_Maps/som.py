# Self Organizing Map向銀行欺騙申請信用卡

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling特徵縮放
from sklearn.preprocessing import MinMaxScaler
#轉成 0 ~ 1 之間(X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))X_scaled = X_std * (max - min) + min)
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom#MiniSom是一種簡約化和Numpy的自組織映射( SOM ) 實現
#MinSom(輸出x大小,輸出y大小,Dataset大小,初始擴展,初始學習率)
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
#初始化權重
som.random_weights_init(X)
#model.train_random(dataset,迭代次數)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()#黑-白
pcolor(som.distance_map().T)#把model置入背景
#setting label
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)# getting the winner
    plot(w[0] + 0.5,#樣本的獲勝位置上標記
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds(欺騙)
mappings = som.win_map(X)
#get frauds id
frauds = np.concatenate((mappings[(3 , 8)], mappings[(2 , 5)]), axis = 0)#winner
#轉換成labels
frauds = sc.inverse_transform(frauds)#id
print(frauds)
