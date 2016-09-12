import numpy as np
import find_mxnet
import mxnet as mx
import logging
import time
import os
from skimage import io, transform
from scipy.stats.stats import pearsonr

logger=logging.getLogger()
logger.setLevel(logging.DEBUG)



# Load the pre-trained model
model = mx.model.FeedForward.load("model_vgg/vgg-0", 218, ctx=mx.gpu(1), numpy_batch_size=1)
mean_img=mx.nd.load("model/mean_224.nd")["mean_img"]

def preImage(path):
    # load image
    img = io.imread(path)
    #print("Original Image Shape: ", img.shape)
    # resize to 224, 224
    resized_img = transform.resize(img, (224, 224))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 255
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    # sub mean
    normed_img = sample - mean_img.asnumpy()
    normed_img.resize(1, 3, 224, 224)
    return normed_img

test="/mnt/ssd/tmp/test/"

pred=np.ndarray(shape=[23817,1584], dtype=np.float32)
index=0
bad=np.zeros(1584)
pre=np.ndarray(shape=23817, dtype=np.dtype((str,64)))
top5=np.ndarray(shape=[23817,5], dtype=np.float32)
top10=np.ndarray(shape=[23817,10], dtype=np.float32)
top15=np.ndarray(shape=[23817,15], dtype=np.float32)
for f in os.listdir(test):
    print(index)
    print(f)
    pre[index]=f
    try:
        image=preImage(os.path.join(test,f))
        pred[index][:]=model.predict(image)[0]
    except:
        pred[index][:]=bad
        print("bad?")
    tmp=np.argsort(pred[index])[::-1]
    top5[index]=tmp[0:5]
    index+=1
with open('tmp_predict_167.csv','w') as f_handle:
    np.savetxt(f_handle, np.c_[pre,pred], delimiter=',', fmt='%s')    

with open('tmp_predict_167_top5.csv','w') as f_handle:
    np.savetxt(f_handle, np.c_[pre,top5], delimiter=',', fmt='%s')
with open('tmp_predict_167_top10.csv','w') as f_handle:
    np.savetxt(f_handle, np.c_[pre,top10], delimiter=',', fmt='%s')
with open('tmp_predict_167_top15.csv','w') as f_handle:
    np.savetxt(f_handle, np.c_[pre,top15], delimiter=',', fmt='%s')

#x=list()
#with open ("/mnt/ssd/tmp/submission_info.csv") as f_handle:
#    for line in f_handle.readlines():
#        x.append(line.replace('\n', '').split(','))

#cr=np.ndarray(shape=[(len(x)-1),2], dtype=np.float32)
#for i in xrange(len(x)-1):
#    print(x[i+1])
#    cr[i][0]=i
#    index1=pre.index(x[i+1][1])
#    index2=pre.index(x[i+1][2])
#    prob1=pred[index1][:]
#    prob2=pred[index2][:]
#    cr[i][1]=pearsonr(prob1,prob2)[0]


#with open('result.csv', 'a') as f_handle:
#    np.savetxt(f_handle, cr, delimiter='\t')



