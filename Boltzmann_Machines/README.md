玻爾茲曼機 Boltzmann Machines
=============================================
<p>非監督學習(unsupervised-learning)</p>
<p>在玻爾茲曼機模型中，不只連接下層神經元，並且連同一層之間的神經元也會連結在一起</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Boltzmann_Machines/220px-Boltzmannexamplev1.png)

限制玻爾茲曼機 Restricted Boltzmann Machines (RBM)
---------------------------------------------

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Boltzmann_Machines/bm0.png)

<p>只有2層結構的淺層神經網路:可視層(visible layer)、隱藏層(hidden layer)</p>
<p>為了降低複雜度，將同一層的神經元彼此間沒有連結，所以稱之"限制"玻爾茲曼機</p>
<p>使用隨機決策(stochastic decisions)決定神經元是否傳導</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Boltzmann_Machines/bm.png)

 <p>前向傳導 (forward):在可視層、每個神經元都會接收到一個資料特徵</p>
 <p>                  b:提高神經元能被激發</p>
 <p>重建(Reconstruction): RBM 的目標是為了重建原始的輸入值 x ，用 a 、相同權重重建成 r</p>
 <p>使用相對熵（relative entropy）又稱為KL散度（Kullback–Leibler divergence)在可視層比較輸入值 x和重建值 r的差異以提高精確度</p>
 
 深度信念網路 Deep Belief Network (DBN)
 -----------------------------------------------
 <p>將 RBM 堆疊起來、建立一個多層的神經網路，成為 DBN</p>
 
 ![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/bm1.png)
 
 <p>預先訓練完後，在最後一層才放「分類器」。無監督或半監督、逐層的預訓練(unsupervised/semi-supervised, layer-wise pre-training)的過程</p>
