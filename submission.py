from scipy.stats.stats import pearsonr
import numpy as np
import math
pred=np.ndarray(shape=[23817,1585], dtype=np.dtype((str,64)))
with open('tmp_predict.csv') as f_handle:
    pred=np.loadtxt(f_handle, delimiter=',', dtype=np.dtype(str,64))


x=list()
with open ("/mnt/ssd/tmp/submission_info.csv") as f_handle:
    for line in f_handle.readlines():
        x.append(line.replace('\n', '').split(','))

#ind=np.array([1], dtype=np.int)
#cr=np.array([1.5], dtype=np.float32)
c=0
with open('result.csv', 'a') as f_handle:
    f_handle.write('index,sameArtist\n')
    for i in range(len(x)-1):
        print(x[i+1])
        index1=np.argwhere(pred==x[i+1][1])[0][0]
        index2=np.argwhere(pred==x[i+1][2])[0][0]
        prob1=pred[index1][1:].astype(float)
        prob2=pred[index2][1:].astype(float)
        try:
            cr=pearsonr(prob1,prob2)[0]
        except:
            cr=-1
        cd=math.exp(cr)/math.exp(1)
        f_handle.write('%d,%f \n'%(c,cd))
        c+=1
        #np.savetxt(f_handle, np.c_[ind,cr], delimiter=',', fmt='%s')


