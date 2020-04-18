import matplotlib.pyplot as plt
import numpy as np 
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = Axes3D(fig)

#find neighbourhood------------interest-----------
def neighbourhood_point(X,x_centroid,dist=3):#鄰近粒子平均值(粒子,重心,距離)
    eligible_X=[]#選擇粒子
    for x in X:#逐一計算距離
        distance_between = distance.euclidean(x,x_centroid)#計算與重心距離
        if distance_between <= dist:#與重心距離小於設定距離
            eligible_X.append(x)#將粒子加入選擇粒子
            
    eligible_X = np.array(eligible_X)#轉為np array
    return eligible_X

#get all neighbourth data
def neighbourhood_data(eligible_X):
    AllPos_Num = np.zeros(4)
    for i in range(len(eligible_X)):
        AllPos_Num[0] = AllPos_Num[0] + eligible_X[i,0]
        AllPos_Num[1] = AllPos_Num[1] + eligible_X[i,1]
        AllPos_Num[2] = AllPos_Num[2] + eligible_X[i,1]
        AllPos_Num[3] = AllPos_Num[3] + 1
    return AllPos_Num

#Generate the data
data_num = 500
data_dim = 3
data = 0 +2*np.random.randn(data_num,data_dim)


#interest center
interest_center = 10*np.random.rand(data_dim)-5



iteration=0
n_center_bt_i_center=100#init distance !=0
while(n_center_bt_i_center!=0):
    #find interest center neighbourhood
    eligible_X = neighbourhood_point(data, interest_center,dist=3)
#    get neighbourhood_data
    AllPos_Num = neighbourhood_data(eligible_X)
    
    n_center_bt_i_center = distance.euclidean(interest_center,[AllPos_Num[0]/AllPos_Num[3],AllPos_Num[1]/AllPos_Num[3],AllPos_Num[2]/AllPos_Num[3]])


    iteration = iteration+1

    interest_center = [AllPos_Num[0]/AllPos_Num[3],AllPos_Num[1]/AllPos_Num[3],AllPos_Num[2]/AllPos_Num[3]]



# plot all data
ax.scatter(data[:,0],data[:,1],data[:,2], s=50, c='b', alpha=0.1, marker='o')
##plot new interest center
ax.scatter(interest_center[0],interest_center[1],interest_center[2],color='k',s=150,alpha=1.0,marker='+')

#plot interest
ax.scatter(eligible_X[:,0],eligible_X[:,1],eligible_X[:,2],color='g',s=50,alpha=0.2)
#plot neighbourth center
ax.scatter(AllPos_Num[0]/AllPos_Num[3],AllPos_Num[1]/AllPos_Num[3],AllPos_Num[2]/AllPos_Num[3],color='r',s=250,alpha=1,marker='*')