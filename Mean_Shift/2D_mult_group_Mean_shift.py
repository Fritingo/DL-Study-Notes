import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from scipy.spatial import distance


c_color = ['r','g','b','c','m','y','k','purple']

#生成data
data_num = 100
data_dim = 2
data = 0 + 2*np.random.randn(data_num, data_dim)
temp = 10 + 3*np.random.randn(data_num, data_dim)
data = np.concatenate((data, temp),axis=0)
temp = 0 + 2*np.random.randn(data_num, data_dim)
temp[:,0] = temp[:,0] + 20
data = np.concatenate((data, temp),axis=0)
temp = 20 + 2*np.random.randn(data_num, data_dim)
temp[:,1] = temp[:,1] 
data = np.concatenate((data, temp),axis=0)
data_num = data_num*4

def neightbourhood_points(X,x_centroid, dist=3):#鄰近粒子平均值(粒子,重心,距離)
    eligible_X = []#選擇粒子
    
    for x in X: #逐一計算距離
        distance_between = distance.euclidean(x, x_centroid)#計算與重心距離
        if distance_between <= dist: #與重心距離小於設定距離
            eligible_X.append(x)#將粒子加入選擇粒子   
            
    eligible_X = np.array(eligible_X)#轉為np array
    mean = np.mean(eligible_X, axis=0)#mean = eligible_X平均值
    return eligible_X, mean#回傳

x = np.copy(data)

iteration = 0 #代數
for i in range(10):#迭代
    mean = np.zeros((data_num,data_dim))#重設mean
    
    for i in range(data_num):#逐一計算粒子
        eligible_X, mean[i] = neightbourhood_points(data, x[i] , dist=5)
        
    plt.clf()#清理畫面
    plt.scatter(data[:,0], data[:,1], color= 'blue',s=50,alpha=0.3)#畫原始粒子
    plt.scatter(x[:,0], x[:,1], color= 'red',s=50,alpha=0.3)#畫鄰近粒子平均值粒子
      
    x = mean
    
    plt.title('iteration' + str(iteration))
    iteration = iteration + 1
    plt.grid()
    plt.show()
    plt.pause(0.1)
    
#重畫group圖
plt.clf()
threshold = 1.0 

center = x[0,:].reshape((1,2))
print(center)
for i in range(data_num):#找重心
    found =False
    for j in range(center.shape[0]):
        dst = distance.euclidean(x[i],center[j])
        
        if dst < threshold:
            found = True
            break
    if not found:
         center = np.concatenate((center,x[i].reshape((1,2))),axis=0)

for center_num in range(len(center)):#畫重心
    plt.scatter(center[center_num,0],center[center_num,1],color=c_color[center_num],s=250,alpha=1,marker='*')

cluster_arr = []

for i in range(data_num):#draw color
        dst_list = []
        for center_num in range(len(center)):#計算粒子與重心距離
            dst = distance.euclidean(center[center_num,:],data[i,:])
            dst_list.append(dst)
        
        cluster = np.argmin(dst_list)#最小值index
        del dst_list[:]
        cluster_arr.append(cluster)
        plt.scatter(data[i,0],data[i,1],color=c_color[cluster],s=50,alpha=0.3)

                 
    
plt.title('iteration' + str(iteration))
plt.grid()
plt.show()