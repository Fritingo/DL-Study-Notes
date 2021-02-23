import numpy  as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


gene_num = 2 #基因數
chromosome_num = 10#染色體數
iteration_num = 5#代數
mutation_rate = 0.3#突變率
crossover_rate = 0.7#交配率
select_ratio = 0.3#選擇比例

select_num = int(chromosome_num * select_ratio)#選擇的染色體數
copy_num =  int(chromosome_num - select_num)#複製的染色體數(被選擇的染色體取代)

#人口[染色體[基因]]
population = np.random.rand(chromosome_num, gene_num)#人口
goal = np.random.rand(gene_num)#目標
best_chromosome = np.array([0,0])#最適合目標的染色體
best_fitness = 0#最適合目標的染色體與目標的適合度

fitness_array = [] #init 適合度arrary
for i in range(chromosome_num):#所有適合度
    fitness = 1.0/ (1.0 + distance.euclidean(population[i,:], goal))#適合度 1/(1+染色體與目標距離)
    fitness_array.append(fitness)#加到適合度arrary
    
for iteration in range(iteration_num):#迭代

    selected_idx = np.argsort(fitness_array)[-select_num:]#挑出最適合的選擇的染色體index np.argsort()排序由小到大index [-select_num:]倒數選擇的染色體數
    temp_population = np.copy(population[selected_idx])#最適合的選擇的染色體
    
    for i in range(copy_num):#複製的染色體數
        sel_chromosome = np.random.randint(0, select_num)#隨機選擇最適合的選擇的染色體index
        copy_chromosome = np.copy(temp_population[sel_chromosome,:].reshape((1,gene_num))) #複製選擇的最適合的染色體 
        temp_population = np.concatenate((temp_population, copy_chromosome), axis=0) #一一被選擇的染色體取代
        
    population = np.copy(temp_population)#將全是最適合的染色體的人口複製
        
    for i in range(chromosome_num):#交配
        if np.random.rand(1) < crossover_rate:#隨機交配 random < 交配率 
            sel_chromosome = np.random.randint(0, chromosome_num)#隨機選擇染色體
            sel_gene = np.random.randint(0, gene_num)#隨機選擇基因
            temp = np.copy(population[i, sel_gene])#先保留本身基因
            population[i,sel_gene] = np.copy(population[sel_chromosome, sel_gene])#本身基因變成選擇染色體的選擇基因 
            population[sel_chromosome, sel_gene] = np.copy(temp)#選擇染色體的選擇基因變成本身基因(temp)
    
    for i in range(chromosome_num):#突變
        if np.random.rand(1) < mutation_rate:#隨機突變 random < 突變率 
            sel_gene = np.random.randint(0, gene_num)#隨機選擇基因
            population[i, sel_gene] = np.random.rand(1)#選擇基因變成隨機
    
    
    fitness_array = [] #適合度arrary
    for i in range(chromosome_num):#所有適合度
        fitness = 1.0/ (1.0 + distance.euclidean(population[i,:], goal))#適合度 1/(1+染色體與目標距離)
        fitness_array.append(fitness)#加到適合度arrary
        
        
    if np.max(fitness_array) > best_fitness:#最高適合度 高於 上次最高適合度
        best_fitness = np.max(fitness_array)#更新最高適合度
        best_idx = np.argmax(fitness_array)#更新最高適合度index
        best_chromosome = np.copy(population[best_idx])#更新最高適合度染色體
        
        
    error = distance.euclidean(best_chromosome, goal)#最高適合度染色體與目標距離
      

    plt.clf()#清理畫面
    plt.scatter(population[:,0], population[:,1], color='blue', s=50, alpha=0.3, marker='o')#畫人口
    plt.scatter(best_chromosome[0], best_chromosome[1], color='green', s=250, alpha=0.7, marker='+')#畫最高適合度染色體
    plt.scatter(goal[0], goal[1], color='red', s=250, alpha=1.0, marker='*')#畫目標
    
    
    plt.title('Iteration: ' + str(iteration) + ', Error: ' + str(error))#畫標題
    plt.xlim(0,1)#x軸上下限
    plt.ylim(0,1)#y軸上下限
    plt.grid()#畫格子
    plt.show()#顯示畫
    plt.pause(0.2)#sleep 0.2