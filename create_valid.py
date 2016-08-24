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
train_folder='/mnt/ssd/tmp/train/'
valid_folder='/mnt/ssd/tmp/valid/'

count=dict()
total=0
new=np.ndarray(shape=[14035,3], dtype=np.dtype((str,64)))
for i in xrange(len(y)):
    if y[i] not in count:
        count[y[i]]=1
        train_img=os.path.join(train_folder, x[i+1][0])
        valid_img=os.path.join(valid_folder, x[i+1][0])
        copyfile(train_img, valid_img)
        new[total][0]=total
        new[total][1]=int(z.index(y[i]))
        new[total][2]=x[i+1][0]
        total+=1
    else:
        count[y[i]]+=1
        if count[y[i]]<10:
            train_img=os.path.join(train_folder, x[i+1][0])
            valid_img=os.path.join(valid_folder, x[i+1][0])
            copyfile(train_img, valid_img)
            new[total][0]=total
            new[total][1]=int(z.index(y[i]))
            new[total][2]=x[i+1][0]
            total+=1
print(len(count))
print(total)





with open('valid.lst', 'a') as f_handle:
    np.savetxt(f_handle, new, delimiter='\t', fmt='%s')
