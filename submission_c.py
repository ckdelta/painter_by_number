from scipy.stats.stats import pearsonr
import numpy as np
import math
from joblib import Parallel, delayed
import multiprocessing

num_cores=multiprocessing.cpu_count()
print(num_cores)

pred=np.ndarray(shape=[23817,6], dtype=np.dtype((str,64)))
with open('tmp_predict_167_top10.csv') as f_handle:
    pred=np.loadtxt(f_handle, delimiter=',', dtype=np.dtype(str,64))


x=list()
with open ("submission_info.csv") as f_handle:
    for line in f_handle.readlines():
        x.append(line.replace('\n', '').split(','))

#ind=np.array([1], dtype=np.int)
#cr=np.array([1.5], dtype=np.float32)
with open('result.csv', 'a') as f_handle:
    f_handle.write('index,sameArtist\n')

def exe(i):
    with open('result.csv', 'a') as f_handle:
        if i%1000 == 0:
            print(x[i+1])
        index1=np.argwhere(pred==x[i+1][1])[0][0]
        index2=np.argwhere(pred==x[i+1][2])[0][0]
        pred1=pred[index1][1:].astype(float)
        pred2=pred[index2][1:].astype(float) 
        cd = 0.0
        if pred1[0] == pred2[0]:
            cd=1.0
        elif len(set(pred1[0:1]).intersection(pred2[0:1])) > 0:
#            print('b')
            cd=0.99
        elif len(set(pred1[0:2]).intersection(pred2[0:2])) > 0:
            cd=0.97
        elif len(set(pred1[0:3]).intersection(pred2[0:3])) > 0: 
            cd=0.95
        elif len(set(pred1[0:4]).intersection(pred2[0:4])) > 0: 
            cd=0.9
        elif len(set(pred1[0:5]).intersection(pred2[0:5])) > 0: 
            cd=0.85
        elif len(set(pred1[0:6]).intersection(pred2[0:6])) > 0: 
            cd=0.8
        elif len(set(pred1[0:7]).intersection(pred2[0:7])) > 0: 
            cd=0.75
        elif len(set(pred1[0:8]).intersection(pred2[0:8])) > 0: 
            cd=0.7
        elif len(set(pred1[0:9]).intersection(pred2[0:9])) > 0: 
            cd=0.65
        elif len(set(pred1).intersection(pred2)) > 0: 
            cd=0.6
        else:
            cd=0.0
        if len(set(pred1[0:5]).intersection(pred2[0:5])) > 2:
            cd=0.99
        if len(set(pred1).intersection(pred2)) > 4:
            cd=0.99
        f_handle.write('%d,%f \n'%(int(x[i+1][0]),cd))

Parallel(n_jobs=num_cores)(delayed(exe)(i)for i in range(len(x)-1))
