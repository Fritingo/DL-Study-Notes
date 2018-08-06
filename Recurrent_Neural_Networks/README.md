回歸神經網路 Recurrent Neural Networks
=============

# intro
-------------
<p>經常用於處理序列相關的問題</p>

## 結構

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Recurrent_Neural_Networks/rnn1.jpg)

<p>每一次hidden layer的output都會被存到memory，之後的hidden layer也會考慮memory內的值，影響下次使用hidden layer的output</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Recurrent_Neural_Networks/rnn2.jpg)

<p>並非三個NN而是同一個NN(使用相同權重不同時間點)可以看到前一個hidden layer的output帶到下一個時間點hidden layer</p>
