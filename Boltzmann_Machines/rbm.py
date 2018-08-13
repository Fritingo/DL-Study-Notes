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
print('test')
# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
#將五星評價換成liked or  not liked
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
#建神經網絡
class RBM():
    def __init__(self, nv, nh):#初始化
        self.W = torch.randn(nh, nv)#權重weight randn隨機 (size)
        self.a = torch.randn(1, nh)#偏差值 bias
        self.b = torch.randn(1, nv)#偏差值 bias
    def sample_h(self, x):#hidden layer
        wx = torch.mm(x, self.W.t())#張量轉置並套用初始權重
        activation = wx + self.a.expand_as(wx)#激活函數用 a 偏差值
        p_h_given_v = torch.sigmoid(activation)#使用sigmoid function 激活函數
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):#visible layer
        wy = torch.mm(y, self.W)#張量轉置並套用初始權重
        activation = wy + self.b.expand_as(wy)#激活函數用 b 偏差值
        p_v_given_h = torch.sigmoid(activation)#使用sigmoid function 激活函數
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):#train更新
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()#更新權重
        self.b += torch.sum((v0 - vk), 0)#更新 a 偏差值
        self.a += torch.sum((ph0 - phk), 0)#更新 b 偏差值
        
#data size
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):#epoch
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):#batch_size
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)#hidden layer 
        for k in range(10):
            _,hk = rbm.sample_h(vk)#hidden layer 
            _,vk = rbm.sample_v(hk)#visible layer
            vk[v0<0] = v0[v0<0]#將不考慮的值濾掉
        phk,_ = rbm.sample_h(vk)#hidden layer 
        rbm.train(v0, vk, ph0, phk)#train
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))#計算 loss
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)#hidden layer 
        _,v = rbm.sample_v(h)#visible layer
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))#計算 loss
        s += 1.
print('test loss: '+str(test_loss/s))