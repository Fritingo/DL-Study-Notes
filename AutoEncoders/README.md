自編碼AutoEncoder
=====================================================
<p>非監督式</p>
<p>用途是用來做資料壓縮</p>
<p>先進行資料壓縮(Encoder)，再解壓縮(Decoder)，比對 input 、 output(重建) 差異做調整</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/AutoEncoders/autoencoder0.PNG)

<p>而hidden layer 就是資料的 feature ，所以 hidden layer 數較少，以達到壓縮資訊的目的</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/AutoEncoders/autoencoder1.gif)

<p>也可以有很多層</p>

![image](https://github.com/cbc106013/DL-Study-Notes/blob/master/AutoEncoders/autoencoder2.png)

Refer
----------------------------------------------------
https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/

https://www.ycc.idv.tw/ml-course-techniques_6.html
