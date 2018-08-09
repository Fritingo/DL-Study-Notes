自組織對映 Self_Organizing_Maps
===================================================

<p>將N維(N-dimension)的資料映射(mapping)到2維(2-dimension)or 1維(1-dimension)維的空間上並且維持資料中的拓撲(topology在連續變化（如拉伸或彎曲，但不包括撕開或黏合）下維持不變的性)特性</p>
<p>通常用作可視化並輔助查看大量數據之間的關係 Ex:</p>
<p>國家生活質量:越左上越好，越右下越差</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Self_Organizing_Maps/som.jpg)
  
![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Self_Organizing_Maps/som1.jpg)

<p>SOM 屬於unsupervised learning(通過自身的訓練，能自動對輸入進行分類)，訓練時採用“競爭學習”的方式(網絡的輸出神經元之間相互競爭以求被激活，每一次只有一個輸出神經元被激活，而其它神經元的狀態被抑制)

<p>SOM是由Input layer和Computation layer構成的兩層網絡(全連接)</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Self_Organizing_Maps/som2.png)

steps
--------------------------------------------------
<p>1)初始化每個節點的權重</p>
<p>2)從訓練數據集中隨機選擇 vector 並將其呈現給lattice(格子,點陣)</p>
<p>3)檢查每個節點以計算哪一個的權重最像輸入vector。獲勝節點通常被稱為最佳匹配單元（BMU）</p>
<p>4)計算BMU附近的半徑</p>
<p>5)調整每個相鄰節點（在步驟4中找到的節點）權重以使它們更像輸入vector。(節點離BMU越近，其權重就越大)</p>
<p>6)從步驟2重複迭代直到收斂</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Self_Organizing_Maps/som3.jpg)

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Self_Organizing_Maps/som4.jpg)

Refer
-------------------------------------

http://www.ai-junkie.com/ann/som/som1.html

https://dotblogs.com.tw/dragon229/2013/02/04/89919

https://glowingpython.blogspot.com/2013/09/self-organizing-maps.html
