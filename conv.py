import numpy as np
pred=np.ndarray(shape=[23817,1585], dtype=np.dtype((str,64)))
top10=np.ndarray(shape=[23817,10], dtype=np.float32)
with open('tmp_predict_167.csv','r') as f_handle:
    pred=np.loadtxt(f_handle, delimiter=',', dtype=np.dtype(str,64))

x=np.ndarray(shape=23817, dtype=np.dtype((str,64)))
y=np.ndarray(shape=[23817,1584], dtype=np.float32)

for index in range(len(pred)):
    x[index]=pred[index][0]
    y[index]=pred[index][1:]

for index in range(len(pred)):
    tmp=np.argsort(y[index])[::-1]
    top10[index]=tmp[0:10]


with open('tmp_predict_167_top10.csv','w') as f_handle:
    np.savetxt(f_handle, np.c_[x,top10], delimiter=',', fmt='%s')
    
