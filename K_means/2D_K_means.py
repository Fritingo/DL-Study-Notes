import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

k = 5
data_num = 20
data_dim = 2

colors = ['r','g','b','c','m']

data =0 + 2*np.random.randn(data_num,data_dim)
temp =0 + 3*np.random.randn(data_num,data_dim)
data = np.concatenate((data,temp),axis=0)

k_point = np.random.rand(k,data_dim)

dis_of_k = np.zeros((data.shape[0],k))
k_group = np.zeros((k,data_dim+1))

for interion in range(10):
    plt.clf()
    error = 0
    for i in range(data.shape[0]):#each point dis k
        for j in range(k):
            dis_of_k[i,j] = distance.euclidean(data[i],k_point[j])
        plt.scatter(data[i,0],data[i,1],color=colors[np.argmin(dis_of_k[i])],s=50,alpha=0.2)
    
    for i in range(data.shape[0]):
        for j in range(k):
            if(np.argmin(dis_of_k[i]) == j):
                k_group[j,0] = k_group[j,0] + data[i,0]
                k_group[j,1] = k_group[j,1] + data[i,1]
                k_group[j,2] = k_group[j,2] + 1
    
    center = np.zeros((k,data_dim))
    
    for i in range(k):#pirnt k  and group center
        plt.scatter(k_point[i,0],k_point[i,1],color=colors[i],s=250,alpha=1,marker='+')
        center[i] = [k_group[i,0]/k_group[i,2],k_group[i,1]/k_group[i,2]]
        plt.scatter(center[i,0],center[i,1],color=colors[i],s=250,alpha=1,marker='*')
        error = error + distance.euclidean(k_point[i,0],center[i,0]) 
    
    k_point = center     
    
    plt.title('interion'+str(interion)+'error '+str(error))
    plt.pause(0.2)