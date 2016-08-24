import numpy as np
import os, os.path

file="/mnt/ssd/tmp/test"

count=len([name for name in os.listdir(file) if os.path.isfile(os.path.join(file, name)) ])


new=np.ndarray(shape=(count, 3), dtype=np.dtype((str,64)))
i=0

for t in os.listdir(file):
    new[i][0]=i
    new[i][1]=0
    new[i][2]=t
    i+=1

print(count)
print(i)


with open('test.lst', 'a') as f_handle:
    np.savetxt(f_handle, new, delimiter='\t', fmt='%s')
