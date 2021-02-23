import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance 
from mpl_toolkits.mplot3d import Axes3D


goal_num = 10
data_dim = 3
par_num = 5
par_dim = goal_num
iter_num = 50

#plot fig
fig = plt.figure(figsize=(6,8))
if (data_dim==2):
    ax1 = fig.add_subplot('211')
elif (data_dim==3):
    ax1 = fig.add_subplot('211',projection='3d')
ax2 = fig.add_subplot(212)

#constant
w = 0.5
c1 = 0.5
c2 = 0.5

goal = np.random.rand(goal_num,data_dim)
start = np.random.rand(data_dim)

class Particle_Class:
    fit = 0
    pb_fit = 0
    x = np.array([.0]*par_dim)
    pb = np.copy(x)
    v = np.array([.0]*par_dim)
    x_path = np.array([.0]*par_dim)
    
    def get_fit(self):
        self.x_path = np.argsort(self.x)
        # print(self.x_path)
        dis = distance.euclidean(start,goal[self.x_path[0]]) 
        for i in range(goal_num-1):
            dis = distance.euclidean(goal[self.x_path[i]],goal[self.x_path[i+1]]) + dis
        self.fit = 1/(1+dis)
            
        if self.fit > self.pb_fit:
            self.pb_fit = self.fit
            self.pb = np.copy(self.x)
               
    
class PSO_Class:
    gb_fit = 0
    gb =  np.array([.0]*par_dim)
    particle = [Particle_Class()for i in range(par_num)]
    gb_path = np.argsort(gb)
    gb_fit_list = []
    def __init__(self):
        self.gb_fit_list.append(0)
        for i in range(par_num):
            self.particle[i].x = np.random.rand(par_dim)
        self.gb = np.copy(self.particle[0].x)
        
    def get_all_fit(self):
        for i in range(par_num):
            self.particle[i].get_fit()
            if self.particle[i].pb_fit > self.gb_fit:
                self.gb_fit = self.particle[i].pb_fit
                self.gb = np.copy(self.particle[i].pb)
                self.gb_path = np.argsort(self.gb)
            
    
    def update(self):
        for i in range(par_num):
            self.particle[i].v = w*self.particle[i].v + c1*np.random.rand(1)*(self.particle[i].pb-self.particle[i].x) \
                                 + c2*np.random.rand(1)*(self.gb-self.particle[i].x)
            self.particle[i].x += self.particle[i].v

        
    def plot(self):
        ax1.cla()
        
        def path(path,color,alpha=1):
            x = [start[0]]
            y = [start[1]]
            for i in range(goal_num):
                x.append(goal[path[i],0])
                y.append(goal[path[i],1])
            ax1.plot(x,y,color=color,linewidth = 1,alpha = alpha)
        
        def path_3d(path,color,alpha=1):
            x = [start[0]]
            y = [start[1]]
            z = [start[2]]
            for i in range(goal_num):
                x.append(goal[path[i],0])
                y.append(goal[path[i],1])
                z.append(goal[path[i],2])
            ax1.plot(x,y,z,color=color,linewidth = 1,alpha = alpha)
        
        if (data_dim==2):
            for i in range(par_num):
                path(self.particle[i].x_path,'gray',0.3)
            path(self.gb_path,'b')
            
            
        elif(data_dim==3):
            for i in range(par_num):
                path_3d(self.particle[i].x_path,'gray',0.3)
            path_3d(self.gb_path,'b')
            
        ax1.scatter(*start,color = 'b',s = 150,alpha = 1,marker = '*')
        for i in range(goal_num):
            ax1.scatter(*goal[i],color = 'r',s = 150,alpha = 1,marker = '*')
            
        
        #learning curve
        ax2.set_title(' fitness '+str(self.gb_fit))
        self.gb_fit_list.append(self.gb_fit)
        ax2.plot(self.gb_fit_list,color='b')
        
        
        ax1.grid()
        # plt.show()
        plt.pause(1)
        
    
    
PSO = PSO_Class()

not_update = 0
last_fit = 0
while (not_update<30):
    if PSO.gb_fit == last_fit:
        not_update += 1
    last_fit = PSO.gb_fit
    PSO.get_all_fit()
    
    PSO.update()

    PSO.plot()

plt.show()