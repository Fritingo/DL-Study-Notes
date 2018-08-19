# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
#sep(分隔符)engine(c or python)encoding(utf-8)
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
#delimiter=sep(Delimiter)
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
#dtype(types)
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
#把data 轉成user、movies相對對應list
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
#轉換成 pytorch type
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):#module
    def __init__(self, ):#定義初始化
        super(SAE, self).__init__()#使用super繼承SAE
        #hidden layer
        self.fc1 = nn.Linear(nb_movies, 20)#使用線性nn.linear(輸入數,節點數)#encoder
        self.fc2 = nn.Linear(20, 10)#encoder
        self.fc3 = nn.Linear(10, 20)#decoder
        self.fc4 = nn.Linear(20, nb_movies)#decoder
        self.activation = nn.Sigmoid()#激活函數Sigmoid
    def forward(self, x):#前向傳導
        x = self.activation(self.fc1(x))#添加每層激活函數
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)#重建元素
        return x
sae = SAE()
criterion = nn.MSELoss()#loss function
optimizer = optim.RMSprop(sae.parameters(), lr = 0.08, weight_decay = 0.5)#優化器

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):#epoch
    #loss 使用均方根誤差
    train_loss = 0
    s = 0.#初始化
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)#創建輸入vector
        target = input.clone()#設定目標(與input一樣)
        if torch.sum(target.data > 0) > 0:#確保每次評價>0可預測
            output = sae(input)#output
            target.require_grad = False#關閉梯度(減少計算)
            output[target == 0] = 0#節省內存
            loss = criterion(output, target)#loss
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)#計算均方根誤差
            loss.backward()#反向傳播
            train_loss += np.sqrt(loss.data[0]*mean_corrector)#計算平方誤差
            s += 1.
            optimizer.step()#優化器
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))#/s算出平均誤差

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user])#.unsqueeze(0)#創建輸入vector
    target = Variable(test_set[id_user])#設定目標(與test input一樣)
    if torch.sum(target.data > 0) > 0:#確保每次評價>0可預測
        output = sae(input)#output
        target.require_grad = False#關閉梯度(減少計算)
        output[target == 0] = 0#節省內存
        loss = criterion(output, target)#loss
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)#計算均方根誤差
        test_loss += np.sqrt(loss.data[0]*mean_corrector)#計算平方誤差
        s += 1.
print('test loss: '+str(test_loss/s))#/s算出平均誤差