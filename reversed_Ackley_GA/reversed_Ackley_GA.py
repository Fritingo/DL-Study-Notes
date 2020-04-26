import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

#set parameter
x_min = y_min = -5
x_max = y_max = 5
z_min = 0
z_max = 15

#plot figure set
fig = plt.figure(figsize=(6,8))
ax1 = fig.add_subplot('211',projection='3d')
ax2 = fig.add_subplot(212)

ax1.set_xlim(x_min,x_max)
ax1.set_ylim(y_min,y_max)
ax1.set_zlim(z_min,z_max)

ax2.set_title('Learning curve')
ax2.set_xlim(0,30)
ax2.set_ylim(0,1)

#=================step1==================畫reversed Ackley surface圖
def reversed_ackley(x,y):
    z = 15 - (-20 * np.exp(-0.2 * (np.sqrt(0.5 * (x ** 2 + y ** 2)))) -
              np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20)
    return z

#set grid
x = np.arange(x_min,x_max,0.5)
y = np.arange(y_min,y_max,0.5)
xv , yv = np.meshgrid(x,y)
zv = reversed_ackley(xv,yv)

#plot surface
ax1.plot_surface(xv,yv,zv,cmap='terrain',alpha=0.2)

#=================step2==================將reversed Ackely最高點設為目標 設置20個隨機點
goal = np.array([x[np.argmax(zv)%zv.shape[0]],y[np.argmax(zv)%zv.shape[0]],zv[int(np.argmax(zv)/zv.shape[0]),np.argmax(zv)%zv.shape[0]]])
#print(goal) #至高點 np.argmax()找最高點索引值(一維) np.argmax(zv)%zv.shape[0]找最高點x y索引值 
            #zv[int(np.argmax(zv)/zv.shape[0]),np.argmax(zv)%zv.shape[0]] 找zv最高點
#set parameter
c_num = 20
g_num = 3

p = np.random.randn(c_num,g_num)
p[:,2] = reversed_ackley(p[:,0],p[:,1])#高z = reversed_ackley(x,y)

#plot p and goal
ax1.scatter(p[:,0],p[:,1],p[:,2],color='C0',s=50,alpha=0.5,marker='o')
ax1.scatter(goal[0],goal[1],goal[2],color = 'C1',s=250,alpha=1,marker='*')

#=================step3==================基因演算法
#set paramter
cros_rate = 0.7
m_rate = 0.3
sel_rate = 0.3
sel_num = int(c_num*sel_rate)

iteration = 0
best_fitness = 0
learning_c = 0
best_p = 0    

while(iteration<30):
    ax1.clear()
    iteration = iteration + 1
    
    p_fitness = np.zeros(c_num)
    for i in range(p.shape[0]):
        p_fitness[i] = 1/(1 + distance.euclidean(goal,p[i]))
        
    if p_fitness[np.argsort(p_fitness)[-1]] > best_fitness:
        best_fitness = p_fitness[np.argsort(p_fitness)[-1]]
        best_pp = p[np.argsort(p_fitness)[-1]]
    
    sel_c = p[np.argsort(p_fitness)[-sel_num:]]
    for i in range(c_num-sel_num):
        p[np.argsort(p_fitness)[i]] = sel_c[np.random.randint(sel_num)]
    
    for i in range(p.shape[0]):#crossover
        if np.random.rand(1) < cros_rate:
            rand_c = np.random.randint(p.shape[0])
            rand_g = np.random.randint(g_num - 1)#z軸為reversed Ackely
            temp = p[i,rand_g]
            p[i,rand_g] = p[rand_c,rand_g]
            p[rand_c,rand_g] = temp
    
    for i in range(p.shape[0]):#m
        if np.random.rand(1) < m_rate:
            rand_g = np.random.randint(g_num - 1)#z軸為reversed Ackely
            p[i,rand_g] = np.random.normal(best_pp[rand_g])
            
    p[:,2] = reversed_ackley(p[:,0],p[:,1])#高z = reversed_ackley(x,y) 
    
    for i in range(p.shape[0]):
        p_fitness[i] = 1/(1 + distance.euclidean(goal,p[i]))
        
    if p_fitness[np.argsort(p_fitness)[-1]] > best_fitness:
        best_fitness = p_fitness[np.argsort(p_fitness)[-1]]
        best_p = p[np.argsort(p_fitness)[-1]]
#=================step4==================畫Learning curve        
        old_learning_c = learning_c
        learning_c = best_fitness
        ax2.plot((iteration - 1,iteration),(old_learning_c,learning_c),color='b')
        ax1.scatter(best_p[0],best_p[1],best_p[2],color='C2',s=250,alpha=1,marker='+')
        print(best_p)
    else:
        ax2.plot((iteration - 1,iteration),(learning_c,learning_c),color='b')
    #plot
    ax1.set_title('iteration '+str(iteration)+' best_fitness '+str(learning_c))
    ax1.plot_surface(xv,yv,zv,cmap='terrain',alpha=0.2)
    ax1.scatter(p[:,0],p[:,1],p[:,2],color='C0',s=50,alpha=0.5,marker='o')
    ax1.scatter(goal[0],goal[1],goal[2],color='C1',s=250,alpha=1,marker='*')
    
    
    plt.pause(0.2)
    plt.draw()
    plt.show()
    
