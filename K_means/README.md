# K means

```flow
st=>start: Start
e=>end: end
op1=>operation: 設定K點
op2=>operation: 計算各粒子與K點距離
op3=>operation: 分類成K群
op4=>operation: 計算各K群中心點(平均值)
cond=>condition: 是否K點等於K群中心點?
para=>parallel: K點向K群中心點移動

st->op1->op2->op3->op4->cond
cond(yes)->e
cond(no)->para

para(path1, top)->op2
```