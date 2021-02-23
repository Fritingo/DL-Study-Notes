import numpy as np
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt


bettle_tentacles_d = 0.04

c = 5

data_dim = 2


goal = np.random.rand(data_dim)
goal = [0.9,0.9]
print('----------',goal)
start = np.random.rand(data_dim)
start = [0.1,0.1]


bettle_step_list = []

bettle_step_list.append(start)


def get_fitness(x,goal):
        dis = distance.euclidean(x,goal) 
#        for i in range(goal_index.shape[0]-1):
#            dis = distance.euclidean(goal[goal_index[i]],goal[goal_index[i+1]]) + dis
        return 1/(1+dis)
    
    
def plt_path(order,color,num):
    x = [start[0]]
    y = [start[1]]
    for i in range(num):
        order[i]
        x.append(order[i][0])
        y.append(order[i][1])
        
    plt.plot(x,y,color = color,linewidth = 1)
    

    
def odor(xl,xr):
    fly_l = distance.euclidean(xl,goal)
    fly_r = distance.euclidean(xr,goal)
    return np.sign(fly_l-fly_r)



def plt_bettle(position,direction):
    
    bettle_xl = np.append(position[0]-bettle_tentacles_d*np.cos(direction),position[1]-bettle_tentacles_d*np.sin(direction))
    bettle_xr = np.append(position[0]+bettle_tentacles_d*np.cos(direction),position[1]+bettle_tentacles_d*np.sin(direction))
    plt.scatter(position[0],position[1],color='g',s=100,alpha=1,marker='o')
#    plt.scatter(bettle_xl[0],bettle_xl[1],color='b',s=30,alpha=1,marker='o')
#    plt.scatter(bettle_xr[0],bettle_xr[1],color='b',s=30,alpha=1,marker='o')
      
    return bettle_xl,bettle_xr
    
        
bettle_d = np.random.rand(1)*360

plt.clf()

b_xl , b_xr = plt_bettle(start,bettle_d)

#    print(bettle_step_list[-1])
fitness = get_fitness(start,goal)
next_step = np.append(bettle_step_list[-1][0]+bettle_tentacles_d*np.cos(bettle_d)*odor(b_xl,b_xr)*c,\
                      bettle_step_list[-1][1]+bettle_tentacles_d*np.sin(bettle_d)*odor(b_xl,b_xr)*c)
#    
bettle_step_list.append(next_step)


best_fitness = 0
fitness_list = []
#for i in range(20):
iteration = 0
while(best_fitness<0.99):
    bettle_tentacles_d = bettle_tentacles_d*0.95 + 0.0005
    c = c*0.95
    iteration = iteration + 1
    
    if(fitness>best_fitness):
        best_fitness = fitness
    fitness_list.append(best_fitness)
#    bettle_d = np.random.rand(1)*360
    
    plt.subplot(211)
    plt.cla()
    for i in range(len(bettle_step_list)):
        plt.scatter(bettle_step_list[i][0],bettle_step_list[i][1],color='b',s=100,alpha=0.7,marker='$\u2648$')
    
    plt_path(bettle_step_list,'gray',len(bettle_step_list))     
    
    
    b_xl , b_xr = plt_bettle(bettle_step_list[-1],bettle_d)
    
#    print(bettle_step_list[-1])
    next_step = np.append(bettle_step_list[-1][0]+bettle_tentacles_d*np.cos(bettle_d)*odor(b_xl,b_xr)*c,\
                          bettle_step_list[-1][1]+bettle_tentacles_d*np.sin(bettle_d)*odor(b_xl,b_xr)*c)
    bettle_d = bettle_d + (90*odor(b_xl,b_xr))
#    print(len(bettle_step_list))
    fitness = get_fitness(next_step,goal)
    bettle_step_list.append(next_step)
    
    
    
    plt.scatter(goal[0],goal[1],color='r',s=250,alpha=0.7,marker='$\u2665$')

    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.grid()
    
    plt.subplot(212)
    plt.cla()
    
    
    plt.plot(fitness_list,label = 'fitness')
    # plt.show()
    plt.subplot(211)
    plt.title('iteration ' + str(iteration)+' fitness '+str(best_fitness))
    
    plt.pause(0.2)

plt.show()