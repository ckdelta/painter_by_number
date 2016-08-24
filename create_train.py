import numpy as np
from shutil import copyfile
import os

x=list()
y=list()
count=0
with open('train_info.csv') as f_handle:
    for line in f_handle.readlines():
        x.append(line.split(','))
        if count > 0:
            y.append(x[count][1])
        count += 1

z=list()
#z=set(y)

for i in xrange(len(y)):
    try:
        z.index(y[i])
    except:
        z.append(y[i])

print(len(z))
print(z[0])


new=np.ndarray(shape=[len(y),3], dtype=np.dtype((str,64)))

for i in xrange(len(y)):
    new[i][0]=i
    new[i][1]=int(z.index(y[i]))
    new[i][2]=x[i+1][0]

with open('train.lst', 'a') as f_handle:
    np.savetxt(f_handle, new, delimiter='\t', fmt='%s')
