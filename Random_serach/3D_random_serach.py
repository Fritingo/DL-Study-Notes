import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection= '3d')

particle_dim = 3 #粒子維度
particle_num = 10 #粒子數
goal = np.random.rand(particle_dim) #目標
particles = np.random.rand(particle_num,particle_dim) #粒子

#init
dst_min = 100#隨機設一較大去比較 距離
min_ind = 100#隨機設一較大去比較 index

for i in range(particle_num):#逐一比較粒子目標距離
    if distance.euclidean(goal,particles[i,:]) < dst_min:#距離小於最小距離
        dst_min = distance.euclidean(goal,particles[i,:])#設距離為最小距離
        min_ind = i#設index為最小距離index

min_p = particles[min_ind,:]#最小距離的粒子

p_error = dst_min #歷代error最小距離
error = dst_min #error 是最小距離
iteration = 0

while(error != 0):#迭代
    iteration = iteration + 1
    ax.clear()
    particles = np.zeros((particle_num,particle_dim))#清空粒子
    #重設粒子
    particles[:,0]=np.random.uniform(min_p[0]-p_error,min_p[0]+p_error,particle_num)#沿者最小距離的粒子x附近隨機設粒子
    particles[:,1]=np.random.uniform(min_p[1]-p_error,min_p[1]+p_error,particle_num)#沿者最小距離的粒子y附近隨機設粒子
    particles[:,2]=np.random.uniform(min_p[2]-p_error,min_p[2]+p_error,particle_num)#沿者最小距離的粒子z附近隨機設粒子
    
    for j in range(particle_num):#逐一比較粒子目標距離
        if distance.euclidean(goal,particles[j,:]) < dst_min:#距離小於最小距離
            dst_min = distance.euclidean(goal,particles[j,:])#設距離為最小距離
            min_ind = j#設index為最小距離index
            
    error = dst_min#error 是最小距離
    
    if error < p_error:#本次error小於歷代error最小距離
        p_error = error#取代歷代error
        min_p = particles[min_ind,:]#設置最小距離的粒子
    
    ax.scatter(particles[:,0], particles[:,1], particles[:,2], s=50, c='b', alpha=0.3, marker='o')#畫粒子
    ax.scatter(min_p[0],min_p[1], min_p[2], s=250, c='g', alpha=0.7, marker='+')#畫最小距離的粒子
    ax.scatter(goal[0], goal[1], goal[2], s=250, c='r', alpha=1.0, marker='*')#畫目標
#    
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)
    
    plt.title('Iteration: ' + str(iteration) + ', Error: ' + str(error))#畫標題
  
    plt.pause(0.2)#sleep 0.2
