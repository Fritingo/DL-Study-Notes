遞歸神經網路 Recurrent Neural Networks
=============

# intro
-------------
<p>經常用於處理序列相關的問題</p>

## 結構

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Recurrent_Neural_Networks/rnn1.jpg)

<p>每一次 hidden layer 的 output 都會被存到 memory，之後的 hidden layer 也會考慮 memory 內的值，影響下次使用 hidden layer 的 output</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Recurrent_Neural_Networks/rnn2.jpg)

<p>並非三個 NN 而是同一個 NN (使用相同權重不同時間點)可以看到前一個 hidden layer 的 output 帶到下一個時間點 hidden layer</p>

### LSTM 長短期記憶模型(Long Short Term Memory Network)

<p>如果有人說 RNN ，通常是指 LSTM ，而不是單純的 RNN</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/Recurrent_Neural_Networks/rnn.jpg)

<p>LSTM 其實就是將 RNN 中 Hidden Layer 的一個神經元，用一個更加複雜的結構替換</p>

<p>LSTM 有 Input Gate , Output Gate , Forget Gate </p>
<p>Input Gate 決定 input 有沒有進到 Memory Block</p>
<p>Forget Gate 決定要不要洗掉原來 Memory 內的值(Forget Gate close Memory = 0)，接者 Memory 的值要再加上通過 Input Gate 的值</p>
<p>Output Gate 決定是否輸出 Memory 經過 hidden layer 的值</p>
