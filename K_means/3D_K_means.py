import matplotlib.pyplot as plt
import numpy as np 
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = Axes3D(fig)

K = 4
data_num = 15
data_dim = 3
c_color = ['r','g','b','c','m','y','k','purple']
data_num = data_num*K

#Generate the data
data = 0 + 2*np.random.randn(data_num,data_dim)#具標準正態分布
temp = 10 + 3*np.random.randn(data_num,data_dim)
data = np.concatenate((data,temp),axis=0)
temp = 0 + 2*np.random.randn(data_num,data_dim)
temp[:,0] = temp[:,0] + 20
data = np.concatenate((data,temp),axis=0)

#randan K point
choose_idx = np.random.randint(0,data_num-1,size=(K,))
center = data[choose_idx]

cluster_arr = []
iteration=0
k_center_dis=100
while(k_center_dis!=0):
    cluster_arr.clear()
    
    AllPos_Num = np.zeros((K,4))
    
    for i in range(data_num):#draw color
        dst_list = []
        for center_num in range(K):
            dst = distance.euclidean(center[center_num,:],data[i,:])
            dst_list.append(dst)
        
        cluster = np.argmin(dst_list)
        del dst_list[:]
        cluster_arr.append(cluster)
        ax.scatter(data[i,0],data[i,1],data[i,2],color=c_color[cluster],s=50,alpha=0.1)
        for center_num in range(K):
            if cluster == center_num:
                AllPos_Num[center_num,0] = AllPos_Num[center_num,0]+data[i,0]
                AllPos_Num[center_num,1] = AllPos_Num[center_num,1]+data[i,1]
                AllPos_Num[center_num,2] = AllPos_Num[center_num,2]+data[i,2]
                AllPos_Num[center_num,3] = AllPos_Num[center_num,3]+1
                
    k_center_dis = 0
    for i in range(K):#draw K and star
        k_center_dis = k_center_dis + distance.euclidean(center[i,:],[AllPos_Num[i,0]/AllPos_Num[i,3],AllPos_Num[i,1]/AllPos_Num[i,3],AllPos_Num[i,2]/AllPos_Num[i,3]])

    for i in range(K):
        center[i,:] = [AllPos_Num[i,0]/AllPos_Num[i,3],AllPos_Num[i,1]/AllPos_Num[i,3],AllPos_Num[i,2]/AllPos_Num[i,3]]
    
    iteration = iteration+1

#ax.scatter(data[:,0],data[:,1],data[:,2],color=c_color[cluster],s=50,alpha=0.3)
for i in range(K):#draw K and star
    ax.scatter(center[i,0],center[i,1],center[i,2],color=c_color[i],s=100,alpha=1,marker='+')
    ax.scatter(AllPos_Num[i,0]/AllPos_Num[i,3],AllPos_Num[i,1]/AllPos_Num[i,3],AllPos_Num[i,2]/AllPos_Num[i,3],color=c_color[i],s=100,alpha=1,marker='*')