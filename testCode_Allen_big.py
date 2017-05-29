import numpy as np
from fitting import fitHSM
import time

download_time = time.time()

train_input=np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_train_input_big.npy')
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_train_output_big.npy')
print "Download complete: Time %s" %(time.time() - download_time)


call_time = time.time()
out = fitHSM(train_input,train_set)
runtime = time.time() - call_time


print "Finish training %s" %(runtime)
np.save("out_Allen_big.npy",out)
print "Saved"
